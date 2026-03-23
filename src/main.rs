use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};
use reqwest::{blocking::Client, header::CONTENT_LENGTH, Method};
use serde::Deserialize;
use std::{
    fmt,
    fs::{self, File},
    io::BufWriter,
    path::Path,
    str::FromStr,
};
use url::Url;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Clone)]
enum Commands {
    /// list model files available for a given HuggingFace repo
    List { repo: RepoId },
    /// download a model from a given HuggingFace repo
    Tug {
        /// 'org/name' specifier for the model repo on HuggingFace
        repo: RepoId,
        /// filename for the desired model to download. exact matches only
        model: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::List { repo } => {
            let manifest = HfFetcher::fetch_manifest(repo)?;
            println!("Found {} files in {}:", &manifest.files.len(), &repo);
            for (i, file) in manifest.files.iter().enumerate() {
                println!("  {}) {file}", i + 1);
            }
            Ok(())
        }
        Commands::Tug { repo, model } => HfFetcher::download_model(repo, model),
    }
}

pub trait Fetcher {
    fn fetch_manifest(repo: &RepoId) -> Result<HfLfsManifest>;
    fn download_model(repo_id: &RepoId, model: &str) -> Result<()>;
}

pub struct HfFetcher;

impl Fetcher for HfFetcher {
    fn fetch_manifest(repo: &RepoId) -> Result<HfLfsManifest> {
        dbg!(repo);
        let models_path = format!("https://huggingface.co/api/models/{repo}");
        dbg!(&models_path);
        let models_json: HfModelsJson = reqwest::blocking::get(models_path)?.json()?;
        let files = models_json
            .siblings
            .into_iter()
            .map(|s| s.rfilename)
            .collect::<Vec<_>>();
        Ok(HfLfsManifest { files })
    }

    fn download_model(repo_id: &RepoId, model: &str) -> Result<()> {
        let resolve_url = resolve_url(repo_id, model);
        let local_path = prepare_local_path(Path::new("."), model)?;
        let file = File::create(local_path)?;
        let mut writer = BufWriter::new(file);

        let c = Client::new();
        let head_response = c.request(Method::HEAD, dbg!(&resolve_url)).send()?;
        let model_size = head_response
            .headers()
            .get(CONTENT_LENGTH)
            .ok_or(anyhow!(
                "Failed to read content-length header for URL {}",
                &resolve_url
            ))?
            .to_str()?
            .parse::<u64>()?;

        println!("Model download size: ~{}", HumanBytes(model_size));

        let progress = ProgressBar::new(model_size);
        progress.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
            .progress_chars("#>-"));

        let mut response = c.request(Method::GET, resolve_url).send()?;

        std::io::copy(&mut response, &mut progress.wrap_write(&mut writer))?;

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepoId(String);

impl RepoId {
    pub fn new(org: &str, repo: &str) -> RepoId {
        RepoId(format!("{org}/{repo}"))
    }

    pub fn parse(input: &str) -> Result<Self> {
        let (org, repo) = input
            .split_once('/')
            .ok_or_else(|| anyhow!("RepoId expects 'org/repo' format, got: '{}'", input))?;
        Ok(Self::new(org, repo))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RepoId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for RepoId {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

#[derive(Clone, Debug)]
pub struct HfLfsManifest {
    files: Vec<String>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct HfFile {
    rfilename: String,
}
#[derive(Clone, Debug, Deserialize)]
pub struct HfModelsJson {
    siblings: Vec<HfFile>,
}

/// Build the HuggingFace resolve URL for a model file.
fn resolve_url(repo_id: &RepoId, model: &str) -> String {
    format!("https://huggingface.co/{repo_id}/resolve/main/{model}")
}

/// Prepare the local file path for download, creating parent directories as needed.
fn prepare_local_path(base: &Path, model: &str) -> Result<std::path::PathBuf> {
    let local_path = base.join(model);
    if let Some(parent) = local_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    Ok(local_path)
}

pub fn fetch_manifest_url(hf_repo_url: &str) -> Result<HfLfsManifest> {
    let repo = repo_id_from_url(hf_repo_url)?;
    dbg!(&repo);
    HfFetcher::fetch_manifest(&repo)
}

pub fn repo_id_from_url(url: &str) -> Result<RepoId> {
    let parsed = Url::parse(url)?;
    let path_parts = parsed
        .path_segments()
        .ok_or_else(|| anyhow!("No path detected"))?
        .collect::<Vec<&str>>();
    if path_parts.len() < 2 {
        return Err(anyhow!("Insufficient path segments"));
    }
    let user_or_org = path_parts[0];
    let repo = path_parts[1];
    Ok(RepoId::new(user_or_org, repo))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- RepoId parsing ---

    #[test]
    fn test_repo_id_parse_valid() {
        let repo = RepoId::parse("org/repo").unwrap();
        assert_eq!(repo.as_str(), "org/repo");
    }

    #[test]
    fn test_repo_id_parse_missing_slash() {
        let err = RepoId::parse("noslash").unwrap_err();
        assert_eq!(
            err.to_string(),
            "RepoId expects 'org/repo' format, got: 'noslash'"
        );
    }

    #[test]
    fn test_repo_id_parse_uses_first_slash_only() {
        let repo = RepoId::parse("org/repo/extra").unwrap();
        assert_eq!(repo.as_str(), "org/repo/extra");
    }

    #[test]
    fn test_repo_id_from_url_with_extra_path_segments() {
        let result = repo_id_from_url("https://huggingface.co/org/repo/tree/main");
        assert_eq!(result.unwrap(), "org/repo".parse().unwrap());
    }

    #[test]
    fn test_repo_id_from_url_without_enough_path_segments() {
        let err = repo_id_from_url("https://huggingface.co").unwrap_err();
        assert_eq!(err.to_string(), "Insufficient path segments");
    }

    // --- resolve URL construction ---

    #[test]
    fn test_resolve_url_flat_filename() {
        let repo: RepoId = "org/repo".parse().unwrap();
        assert_eq!(
            resolve_url(&repo, "model.gguf"),
            "https://huggingface.co/org/repo/resolve/main/model.gguf"
        );
    }

    #[test]
    fn test_resolve_url_subdirectory_path() {
        let repo: RepoId = "org/repo".parse().unwrap();
        assert_eq!(
            resolve_url(&repo, "subdir/model.safetensors"),
            "https://huggingface.co/org/repo/resolve/main/subdir/model.safetensors"
        );
    }

    #[test]
    fn test_resolve_url_deeply_nested_path() {
        let repo: RepoId = "org/repo".parse().unwrap();
        assert_eq!(
            resolve_url(&repo, "a/b/c/model.bin"),
            "https://huggingface.co/org/repo/resolve/main/a/b/c/model.bin"
        );
    }

    // --- local path preparation ---

    #[test]
    fn test_prepare_local_path_flat_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = prepare_local_path(tmp.path(), "model.gguf").unwrap();
        assert_eq!(path, tmp.path().join("model.gguf"));
        // no subdirectory should be created
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);
    }

    #[test]
    fn test_prepare_local_path_creates_parent_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let path = prepare_local_path(tmp.path(), "subdir/model.safetensors").unwrap();
        assert_eq!(path, tmp.path().join("subdir/model.safetensors"));
        assert!(tmp.path().join("subdir").is_dir());
    }

    #[test]
    fn test_prepare_local_path_creates_deeply_nested_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let path = prepare_local_path(tmp.path(), "a/b/c/model.bin").unwrap();
        assert_eq!(path, tmp.path().join("a/b/c/model.bin"));
        assert!(tmp.path().join("a/b/c").is_dir());
    }

    #[test]
    fn test_prepare_local_path_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let path1 = prepare_local_path(tmp.path(), "subdir/model.bin").unwrap();
        let path2 = prepare_local_path(tmp.path(), "subdir/model.bin").unwrap();
        assert_eq!(path1, path2);
    }

    // --- smoke tests (hit live HuggingFace API, skip in CI) ---

    #[test]
    #[ignore = "hits live HuggingFace API"]
    fn smoke_test_fetch_manifest() {
        let result = HfFetcher::fetch_manifest(&"bert-base-uncased".parse().unwrap());
        let files = result.unwrap().files;
        assert!(!files.is_empty(), "expected manifest to contain files");
    }

    // TODO: mocked integration tests
    //
    // Use a lightweight HTTP server (e.g. wiremock, httpmock, or mockito) to:
    //
    // 1. Mock the HF models API (`/api/models/{repo}`) returning a canned
    //    siblings JSON payload, and verify fetch_manifest deserializes it.
    //
    // 2. Mock the resolve endpoint (`/{repo}/resolve/main/{file}`) returning
    //    a small payload with a Content-Length header, and verify download_model
    //    writes the correct bytes to the expected local path — including
    //    subdirectory paths.
    //
    // This would let us test the full download flow without network access
    // or multi-GB files.
}
