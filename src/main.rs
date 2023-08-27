use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};
use reqwest::{blocking::Client, header::CONTENT_LENGTH, Method};
use serde::Deserialize;
use std::{fmt, fs::File, io::BufWriter, str::FromStr};
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
        let resolve_url = format!("https://huggingface.co/{repo_id}/resolve/main/{model}");
        let file = File::create(model)?;
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

    #[test]
    fn test_download_model() {
        let model_filename = "llongorca-7b-16k.ggmlv3.q5_K_M.bin";
        let result = HfFetcher::download_model(
            &"TheBloke/LlongOrca-7B-16K-GGML".parse().unwrap(),
            model_filename,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_fetch_manifest() {
        // start mock server
        // set up test url constructed for mock server
        // call manifest fn with test url
        let result = HfFetcher::fetch_manifest(&"TheBloke/LlongOrca-7B-16K-GGML".parse().unwrap());
        let files = result.unwrap().files;
        assert_eq!(files, Vec::<String>::new());
    }

    #[test]
    fn test_fetch_manifest_url() {
        // start mock server
        // set up test url constructed for mock server
        // call manifest fn with test url
        let result = fetch_manifest_url("https://huggingface.co/TheBloke/LlongOrca-7B-16K-GGML");
        let files = result.unwrap().files;
        assert_eq!(files, Vec::<String>::new());
    }

    #[test]
    fn test_repo_id_from_url_with_extra_path_segments() {
        let result = repo_id_from_url("https://huggingface.co/org/repo/tree/main");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "org/repo".parse().unwrap());
    }

    #[test]
    fn test_repo_id_from_url_without_enough_path_segments() {
        let result = repo_id_from_url("https://huggingface.co");
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Insufficient path segments"
        );
    }
}
