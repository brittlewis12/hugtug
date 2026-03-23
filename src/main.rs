use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};
use reqwest::{blocking::Client, header::CONTENT_LENGTH, Method};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fmt,
    fs::{self, File},
    io::BufWriter,
    path::Path,
    str::FromStr,
};
use url::Url;

// ── CLI ──────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Clone)]
enum Commands {
    /// search for models on HuggingFace
    Search {
        /// search query (e.g. "llama 3 8b gguf")
        query: String,
        /// max number of results
        #[arg(short, long, default_value_t = 10)]
        limit: u32,
        /// sort by trending score instead of downloads
        #[arg(long)]
        trending: bool,
        /// show file list with sizes for each result
        #[arg(long)]
        files: bool,
        /// include all files (e.g. .gitattributes)
        #[arg(short, long)]
        all: bool,
    },
    /// list model files available for a given HuggingFace repo
    List {
        /// 'org/name' specifier or full URL for the model repo
        repo: RepoId,
        /// include all files (e.g. .gitattributes)
        #[arg(short, long)]
        all: bool,
    },
    /// download a model from a given HuggingFace repo
    Tug {
        /// 'org/name' specifier for the model repo on HuggingFace
        repo: RepoId,
        /// filename for the desired model to download. exact matches only
        model: String,
    },
}

// ── Commands ─────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Search {
            query,
            limit,
            trending,
            files,
            all,
        } => cmd_search(&query, limit, trending, files, all),
        Commands::List { repo, all } => cmd_list(&repo, all),
        Commands::Tug { repo, model } => cmd_download(&repo, &model),
    }
}

fn cmd_search(query: &str, limit: u32, trending: bool, show_files: bool, all: bool) -> Result<()> {
    let sort = if trending {
        "trendingScore"
    } else {
        "downloads"
    };
    let client = Client::new();
    let SearchResults { repos, total } = search_models(&client, query, limit, sort)?;

    if repos.is_empty() {
        println!("No repos found matching \"{query}\"");
        return Ok(());
    }

    match total {
        Some(t) if t > repos.len() as u64 => {
            println!(
                "Showing {} of {} repos matching \"{query}\":\n",
                repos.len(),
                format_count(t)
            );
        }
        _ => println!("Found {} repos matching \"{query}\":\n", repos.len()),
    }

    for (i, result) in repos.iter().enumerate() {
        let ext_summary = format_ext_breakdown(
            result
                .siblings
                .iter()
                .map(|s| s.rfilename.as_str())
                .filter(|f| all || is_visible(f)),
        );
        println!("  {}) {}", i + 1, &result.id);
        println!(
            "     {} {}",
            format_downloads(result.downloads),
            ext_summary
        );

        if show_files {
            let files = fetch_repo_files(&client, &result.id, all)?;
            let name_width = files.iter().map(|f| f.path.len()).max().unwrap_or(0);
            for fi in &files {
                let size_str = fi
                    .size
                    .map(|s| HumanBytes(s).to_string())
                    .unwrap_or_default();
                println!(
                    "     -> {:<width$}  {}",
                    fi.path,
                    size_str,
                    width = name_width
                );
            }
        }
    }
    Ok(())
}

fn cmd_list(repo: &RepoId, all: bool) -> Result<()> {
    let client = Client::new();
    let files = fetch_repo_files(&client, repo.as_str(), all)?;
    let name_width = files.iter().map(|f| f.path.len()).max().unwrap_or(0);

    println!("Found {} files in {}:", files.len(), repo);
    for (i, fi) in files.iter().enumerate() {
        let size_str = fi
            .size
            .map(|s| HumanBytes(s).to_string())
            .unwrap_or_default();
        println!(
            "  {:>2}) {:<width$}  {}",
            i + 1,
            fi.path,
            size_str,
            width = name_width
        );
    }
    Ok(())
}

fn cmd_download(repo_id: &RepoId, model: &str) -> Result<()> {
    let url = resolve_url(repo_id, model);
    let local_path = prepare_local_path(Path::new("."), model)?;
    let file = File::create(local_path)?;
    let mut writer = BufWriter::new(file);

    let client = Client::new();
    let head_response = client.request(Method::HEAD, &url).send()?;
    let model_size = head_response
        .headers()
        .get(CONTENT_LENGTH)
        .ok_or(anyhow!(
            "Failed to read content-length header for URL {}",
            &url
        ))?
        .to_str()?
        .parse::<u64>()?;

    println!("Model download size: ~{}", HumanBytes(model_size));

    let progress = ProgressBar::new(model_size);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
            .progress_chars("#>-"),
    );

    let mut response = client.request(Method::GET, url).send()?;
    std::io::copy(&mut response, &mut progress.wrap_write(&mut writer))?;

    Ok(())
}

/// Files that are git/repo plumbing and never useful to download.
const HIDDEN_FILES: &[&str] = &[".gitattributes"];

fn is_visible(filename: &str) -> bool {
    !HIDDEN_FILES.contains(&filename)
}

// ── API ──────────────────────────────────────────────────────────────

/// Results from a HuggingFace model search.
struct SearchResults {
    repos: Vec<HfSearchResult>,
    total: Option<u64>,
}

/// Search HuggingFace models, returning results with file lists and download counts.
fn search_models(client: &Client, query: &str, limit: u32, sort: &str) -> Result<SearchResults> {
    let response = client
        .get("https://huggingface.co/api/models")
        .query(&[
            ("search", query),
            ("sort", sort),
            ("direction", "-1"),
            ("limit", &limit.to_string()),
            ("expand[]", "siblings"),
            ("expand[]", "downloads"),
        ])
        .send()?;
    let repos: Vec<HfSearchResult> = response.json()?;
    let total = fetch_model_count(client, query);
    Ok(SearchResults { repos, total })
}

/// Fetch total model count for a query via the models-json endpoint.
fn fetch_model_count(client: &Client, query: &str) -> Option<u64> {
    let response = client
        .get("https://huggingface.co/models-json")
        .query(&[("search", query), ("withCount", "true")])
        .send()
        .ok()?;
    let data: ModelsCountResponse = response.json().ok()?;
    Some(data.num_total_items)
}

#[derive(Deserialize)]
struct ModelsCountResponse {
    #[serde(rename = "numTotalItems")]
    num_total_items: u64,
}

/// Fetch file list with sizes for a single repo (uses ?blobs=true).
fn fetch_repo_files(client: &Client, repo_id: &str, all: bool) -> Result<Vec<FileInfo>> {
    let url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");
    let response: HfModelsJson = client.get(&url).send()?.json()?;
    Ok(response
        .siblings
        .into_iter()
        .filter(|s| all || is_visible(&s.rfilename))
        .map(|s| FileInfo {
            path: s.rfilename,
            size: s.size,
        })
        .collect())
}

// ── Formatting ───────────────────────────────────────────────────────

/// Format a large count with comma separators.
fn format_count(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Format a download count as a compact human-readable string.
fn format_downloads(n: u64) -> String {
    match n {
        0..1_000 => format!("\u{2193} {n}"),
        1_000..10_000 => format!("\u{2193} {:.1}K", n as f64 / 1_000.0),
        10_000..1_000_000 => format!("\u{2193} {:.0}K", n as f64 / 1_000.0),
        _ => format!("\u{2193} {:.1}M", n as f64 / 1_000_000.0),
    }
}

/// Summarize file extensions with counts, sorted by frequency.
/// Example: "18 gguf  2 json  1 md"
fn format_ext_breakdown<'a>(filenames: impl Iterator<Item = &'a str>) -> String {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for name in filenames {
        let ext = Path::new(name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("other");
        *counts.entry(ext.to_string()).or_default() += 1;
    }
    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let parts: Vec<String> = sorted.iter().map(|(ext, n)| format!("{n} {ext}")).collect();
    parts.join("  ")
}

// ── Helpers ──────────────────────────────────────────────────────────

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

// ── Types ────────────────────────────────────────────────────────────

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

/// A file entry returned by the HuggingFace API.
#[derive(Clone, Debug, Deserialize)]
pub struct HfFile {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

/// Response from /api/models/{repo} (single model detail).
#[derive(Clone, Debug, Deserialize)]
pub struct HfModelsJson {
    siblings: Vec<HfFile>,
}

/// A single result from the /api/models search endpoint.
#[derive(Clone, Debug, Deserialize)]
pub struct HfSearchResult {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    siblings: Vec<HfFile>,
}

/// Processed file info for display.
pub struct FileInfo {
    path: String,
    size: Option<u64>,
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- format_downloads ---

    #[test]
    fn test_format_downloads_zero() {
        assert_eq!(format_downloads(0), "\u{2193} 0");
    }

    #[test]
    fn test_format_downloads_small() {
        assert_eq!(format_downloads(42), "\u{2193} 42");
        assert_eq!(format_downloads(999), "\u{2193} 999");
    }

    #[test]
    fn test_format_downloads_thousands_with_decimal() {
        assert_eq!(format_downloads(1_000), "\u{2193} 1.0K");
        assert_eq!(format_downloads(1_500), "\u{2193} 1.5K");
        assert_eq!(format_downloads(9_999), "\u{2193} 10.0K");
    }

    #[test]
    fn test_format_downloads_thousands_no_decimal() {
        assert_eq!(format_downloads(10_000), "\u{2193} 10K");
        assert_eq!(format_downloads(268_525), "\u{2193} 269K");
        assert_eq!(format_downloads(999_999), "\u{2193} 1000K");
    }

    #[test]
    fn test_format_downloads_millions() {
        assert_eq!(format_downloads(1_000_000), "\u{2193} 1.0M");
        assert_eq!(format_downloads(2_100_000), "\u{2193} 2.1M");
        assert_eq!(format_downloads(7_561_380), "\u{2193} 7.6M");
    }

    // --- format_ext_breakdown ---

    #[test]
    fn test_ext_breakdown_single_type() {
        let files = vec!["model.gguf", "model2.gguf"];
        assert_eq!(format_ext_breakdown(files.into_iter()), "2 gguf");
    }

    #[test]
    fn test_ext_breakdown_multiple_types_sorted_by_count() {
        let files = vec!["a.gguf", "b.gguf", "c.gguf", "config.json", "README.md"];
        assert_eq!(
            format_ext_breakdown(files.into_iter()),
            "3 gguf  1 json  1 md"
        );
    }

    #[test]
    fn test_ext_breakdown_no_extension() {
        let files = vec!["LICENSE", "a.gguf"];
        assert_eq!(format_ext_breakdown(files.into_iter()), "1 gguf  1 other");
    }

    #[test]
    fn test_ext_breakdown_empty() {
        let files: Vec<&str> = vec![];
        assert_eq!(format_ext_breakdown(files.into_iter()), "");
    }

    #[test]
    fn test_ext_breakdown_same_count_alpha_sorted() {
        let files = vec!["a.safetensors", "b.json", "c.gguf"];
        // all count=1, should be alphabetical
        assert_eq!(
            format_ext_breakdown(files.into_iter()),
            "1 gguf  1 json  1 safetensors"
        );
    }

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

    // --- deserialization ---

    #[test]
    fn test_deserialize_hf_file_without_size() {
        let json = r#"{"rfilename": "model.gguf"}"#;
        let file: HfFile = serde_json::from_str(json).unwrap();
        assert_eq!(file.rfilename, "model.gguf");
        assert_eq!(file.size, None);
    }

    #[test]
    fn test_deserialize_hf_file_with_size() {
        let json = r#"{"rfilename": "model.gguf", "size": 4081004224}"#;
        let file: HfFile = serde_json::from_str(json).unwrap();
        assert_eq!(file.rfilename, "model.gguf");
        assert_eq!(file.size, Some(4_081_004_224));
    }

    #[test]
    fn test_deserialize_search_result() {
        let json = r#"{
            "id": "org/repo",
            "downloads": 268525,
            "siblings": [
                {"rfilename": "model.gguf"},
                {"rfilename": "config.json"}
            ]
        }"#;
        let result: HfSearchResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.id, "org/repo");
        assert_eq!(result.downloads, 268_525);
        assert_eq!(result.siblings.len(), 2);
    }

    #[test]
    fn test_deserialize_search_result_minimal() {
        let json = r#"{"id": "org/repo"}"#;
        let result: HfSearchResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.id, "org/repo");
        assert_eq!(result.downloads, 0);
        assert!(result.siblings.is_empty());
    }

    // --- smoke tests (hit live HuggingFace API, skip in CI) ---

    #[test]
    #[ignore = "hits live HuggingFace API"]
    fn smoke_test_search() {
        let client = Client::new();
        let SearchResults { repos, total } =
            search_models(&client, "llama gguf", 3, "downloads").unwrap();
        assert!(!repos.is_empty(), "expected search to return results");
        assert!(repos[0].downloads > 0, "expected download counts");
        assert!(
            !repos[0].siblings.is_empty(),
            "expected siblings in results"
        );
        assert!(
            total.is_some_and(|t| t > 0),
            "expected total model count from quicksearch"
        );
    }

    #[test]
    #[ignore = "hits live HuggingFace API"]
    fn smoke_test_list_with_sizes() {
        let client = Client::new();
        let files = fetch_repo_files(&client, "bert-base-uncased", false).unwrap();
        assert!(!files.is_empty(), "expected files in repo");
        // at least some files should have sizes
        assert!(
            files.iter().any(|f| f.size.is_some()),
            "expected some files to have sizes"
        );
    }

    // TODO: mocked integration tests
    //
    // Use a lightweight HTTP server (e.g. wiremock, httpmock, or mockito) to:
    //
    // 1. Mock the HF models API (`/api/models/{repo}`) returning a canned
    //    siblings JSON payload, and verify fetch_repo_files deserializes it.
    //
    // 2. Mock the search endpoint (`/api/models?search=...`) returning canned
    //    results and verify search_models parses them correctly.
    //
    // 3. Mock the resolve endpoint (`/{repo}/resolve/main/{file}`) returning
    //    a small payload with a Content-Length header, and verify cmd_download
    //    writes the correct bytes to the expected local path.
}
