use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use fs2::FileExt;
use indicatif::{HumanBytes, ProgressBar, ProgressStyle};
use reqwest::{
    Method, StatusCode,
    blocking::{Client, Response},
    header::{ACCEPT_ENCODING, CONTENT_ENCODING, HeaderMap, RANGE},
    redirect::Policy,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fmt,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    str::FromStr,
    time::{SystemTime, UNIX_EPOCH},
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
        /// directory where the repo-relative model path should be written
        #[arg(short = 'd', long = "dir", alias = "output-dir", default_value = ".")]
        dir: PathBuf,
        /// replace an existing local file if it differs from the remote file
        #[arg(long)]
        force: bool,
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
        Commands::Tug {
            repo,
            model,
            dir,
            force,
        } => cmd_download(&repo, &model, &dir, force),
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

        let mut age_parts: Vec<String> = Vec::new();
        if let Some(age) = result.created_at.as_deref().and_then(format_age) {
            age_parts.push(format!("created {age}"));
        }
        if let Some(age) = result.last_modified.as_deref().and_then(format_age) {
            age_parts.push(format!("updated {age}"));
        }
        let age_str = if age_parts.is_empty() {
            String::new()
        } else {
            format!("  \u{b7}  {}", age_parts.join("  \u{b7}  "))
        };

        println!("  {}) {}", i + 1, &result.id);
        println!(
            "     {} {}{}",
            format_downloads(result.downloads),
            ext_summary,
            age_str,
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

fn cmd_download(repo_id: &RepoId, model: &str, output_dir: &Path, force: bool) -> Result<()> {
    let remote_path = RemotePath::parse(model)?;
    let client = Client::new();
    let remote = probe_remote(&client, repo_id, &remote_path)?;
    let paths = download_paths(output_dir, &remote_path)?;

    ensure_download_dirs(&paths)?;
    let lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&paths.lock_path)
        .with_context(|| format!("Failed to open lock file {}", paths.lock_path.display()))?;
    lock_file.lock_exclusive().with_context(|| {
        format!(
            "Failed to acquire download lock {}",
            paths.lock_path.display()
        )
    })?;

    download_locked(&client, &paths, &remote, force)
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
            ("expand[]", "lastModified"),
            ("expand[]", "createdAt"),
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

/// Convert a civil date to days since Unix epoch (1970-01-01).
/// Algorithm from <http://howardhinnant.github.io/date_algorithms.html>.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719_468
}

/// Parse an ISO 8601 UTC timestamp ("2026-05-06T10:02:17.000Z") to Unix epoch seconds.
fn parse_iso8601_epoch_secs(s: &str) -> Option<u64> {
    if s.len() < 19 {
        return None;
    }
    let year: i64 = s.get(0..4)?.parse().ok()?;
    let month: u32 = s.get(5..7)?.parse().ok()?;
    let day: u32 = s.get(8..10)?.parse().ok()?;
    let hour: u64 = s.get(11..13)?.parse().ok()?;
    let min: u64 = s.get(14..16)?.parse().ok()?;
    let sec: u64 = s.get(17..19)?.parse().ok()?;
    let days = days_from_civil(year, month, day);
    if days < 0 {
        return None;
    }
    Some(days as u64 * 86_400 + hour * 3600 + min * 60 + sec)
}

/// Format an ISO 8601 timestamp as a compact relative-time string (e.g. "18d ago", "3mo ago").
fn format_age(iso: &str) -> Option<String> {
    let then = parse_iso8601_epoch_secs(iso)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    let delta = now.saturating_sub(then);
    let s = match delta {
        0..60 => "just now".to_string(),
        60..3_600 => format!("{}m ago", delta / 60),
        3_600..86_400 => format!("{}h ago", delta / 3_600),
        86_400..2_592_000 => format!("{}d ago", delta / 86_400),
        2_592_000..31_536_000 => format!("{}mo ago", delta / 2_592_000),
        _ => format!("{}y ago", delta / 31_536_000),
    };
    Some(s)
}

// ── Helpers ──────────────────────────────────────────────────────────

const DEFAULT_REVISION: &str = "main";
const RESUME_SCHEMA_VERSION: u8 = 1;

/// Build the HuggingFace resolve URL for a model file.
#[cfg_attr(not(test), allow(dead_code))]
fn resolve_url(repo_id: &RepoId, model: &str) -> String {
    let remote_path = RemotePath::parse(model).expect("invalid remote path");
    resolve_url_for_revision(repo_id, DEFAULT_REVISION, &remote_path).expect("invalid resolve URL")
}

fn resolve_url_for_revision(
    repo_id: &RepoId,
    revision: &str,
    remote_path: &RemotePath,
) -> Result<String> {
    let mut url = Url::parse("https://huggingface.co")?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow!("Failed to build HuggingFace URL"))?;
        for segment in repo_id.as_str().split('/') {
            segments.push(segment);
        }
        segments.push("resolve");
        segments.push(revision);
        for segment in &remote_path.components {
            segments.push(segment);
        }
    }
    Ok(url.to_string())
}

/// Prepare the local file path for download, creating parent directories as needed.
#[cfg_attr(not(test), allow(dead_code))]
fn prepare_local_path(base: &Path, model: &str) -> Result<PathBuf> {
    let remote_path = RemotePath::parse(model)?;
    let local_path = remote_path.join_under(base);
    if let Some(parent) = local_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    Ok(local_path)
}

fn probe_remote(
    client: &Client,
    repo_id: &RepoId,
    remote_path: &RemotePath,
) -> Result<RemoteIdentity> {
    let url = resolve_url_for_revision(repo_id, DEFAULT_REVISION, remote_path)?;
    let probe_client = Client::builder().redirect(Policy::none()).build()?;
    let response = probe_client
        .request(Method::HEAD, &url)
        .header(ACCEPT_ENCODING, "identity")
        .send()
        .with_context(|| format!("Failed to probe remote URL {url}"))?;
    let status = response.status();
    if !(status.is_success() || status.is_redirection()) {
        bail!("Remote probe failed with HTTP {status} for {url}");
    }

    let headers = response.headers().clone();
    let fallback_headers = if header_value(&headers, "X-Repo-Commit").is_none()
        || remote_size_from_headers(&headers, status).is_none()
    {
        Some(range_probe(client, &url)?)
    } else {
        None
    };

    let commit = header_value(&headers, "X-Repo-Commit")
        .or_else(|| {
            fallback_headers
                .as_ref()
                .and_then(|h| header_value(h, "X-Repo-Commit"))
        })
        .ok_or_else(|| anyhow!("Remote response did not include X-Repo-Commit for {url}"))?;
    let expected_size = remote_size_from_headers(&headers, status)
        .or_else(|| {
            fallback_headers
                .as_ref()
                .and_then(|h| remote_size_from_headers(h, StatusCode::PARTIAL_CONTENT))
        })
        .ok_or_else(|| anyhow!("Remote response did not include a usable size for {url}"))?;
    let commit_url = resolve_url_for_revision(repo_id, &commit, remote_path)?;

    Ok(RemoteIdentity {
        repo_id: repo_id.as_str().to_string(),
        path: remote_path.as_str().to_string(),
        revision: DEFAULT_REVISION.to_string(),
        resolved_commit: commit,
        url: commit_url,
        expected_size,
        etag: header_value(&headers, "ETag").or_else(|| {
            fallback_headers
                .as_ref()
                .and_then(|h| header_value(h, "ETag"))
        }),
        linked_etag: header_value(&headers, "X-Linked-ETag").or_else(|| {
            fallback_headers
                .as_ref()
                .and_then(|h| header_value(h, "X-Linked-ETag"))
        }),
        xet_hash: header_value(&headers, "X-Xet-Hash").or_else(|| {
            fallback_headers
                .as_ref()
                .and_then(|h| header_value(h, "X-Xet-Hash"))
        }),
        accept_ranges: header_value(&headers, "Accept-Ranges")
            .or_else(|| {
                fallback_headers
                    .as_ref()
                    .and_then(|h| header_value(h, "Accept-Ranges"))
            })
            .is_some_and(|v| v.eq_ignore_ascii_case("bytes")),
    })
}

fn range_probe(client: &Client, url: &str) -> Result<HeaderMap> {
    let response = client
        .request(Method::GET, url)
        .header(RANGE, "bytes=0-0")
        .header(ACCEPT_ENCODING, "identity")
        .send()
        .with_context(|| format!("Failed to range-probe remote URL {url}"))?;
    let status = response.status();
    if !(status.is_success() || status == StatusCode::PARTIAL_CONTENT) {
        bail!("Remote range probe failed with HTTP {status} for {url}");
    }
    Ok(response.headers().clone())
}

fn download_paths(output_dir: &Path, remote_path: &RemotePath) -> Result<DownloadPaths> {
    let final_path = remote_path.join_under(output_dir);
    let final_name = final_path
        .file_name()
        .ok_or_else(|| anyhow!("Remote path did not include a file name"))?
        .to_string_lossy()
        .to_string();
    let parent = final_path.parent().unwrap_or_else(|| Path::new("."));
    let state_dir = parent.join(".hugtug");
    Ok(DownloadPaths {
        final_path,
        part_path: state_dir.join(format!("{final_name}.part")),
        metadata_path: state_dir.join(format!("{final_name}.json")),
        metadata_tmp_path: state_dir.join(format!("{final_name}.json.tmp")),
        failed_part_path: state_dir.join(format!("{final_name}.failed-sha256.part")),
        failed_metadata_path: state_dir.join(format!("{final_name}.failed-sha256.json")),
        failed_metadata_tmp_path: state_dir.join(format!("{final_name}.failed-sha256.json.tmp")),
        lock_path: state_dir.join(format!("{final_name}.lock")),
        state_dir,
    })
}

fn ensure_download_dirs(paths: &DownloadPaths) -> Result<()> {
    if let Some(parent) = paths
        .final_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }
    fs::create_dir_all(&paths.state_dir).with_context(|| {
        format!(
            "Failed to create hugtug state directory {}",
            paths.state_dir.display()
        )
    })?;
    Ok(())
}

fn download_locked(
    client: &Client,
    paths: &DownloadPaths,
    remote: &RemoteIdentity,
    force: bool,
) -> Result<()> {
    remove_file_if_exists(&paths.metadata_tmp_path)?;
    let metadata = match read_resume_metadata(&paths.metadata_path) {
        Ok(metadata) => metadata,
        Err(_) => None,
    };

    if paths.final_path.exists() {
        let stat = fs::metadata(&paths.final_path)
            .with_context(|| format!("Failed to inspect {}", paths.final_path.display()))?;
        if !stat.is_file() {
            bail!("{} exists but is not a file", paths.final_path.display());
        }
    }

    if paths.final_path.exists() && !force {
        let final_match = final_matches_remote(paths, remote, metadata.as_ref())?;
        if final_match.matches {
            remove_file_if_exists(&paths.part_path)?;
            write_completed_metadata(paths, remote, final_match.sha256, metadata.as_ref())?;
            println!("Already downloaded: {}", paths.final_path.display());
            return Ok(());
        }
        bail!(
            "{} already exists but does not match the remote file; pass --force to replace it",
            paths.final_path.display()
        );
    }

    let resume_metadata = reconcile_partial_state(paths, remote, metadata)?;
    let created_at = resume_metadata
        .as_ref()
        .map(|m| m.created_at)
        .unwrap_or_else(now_secs);
    write_resume_metadata_atomic(
        paths,
        &ResumeMetadata::new(remote.clone(), paths, created_at, None),
    )?;

    println!("Model download size: ~{}", HumanBytes(remote.expected_size));
    let verified_hash = stream_fresh_or_resumed(client, paths, remote)?;
    finalize_download(paths, remote, verified_hash)?;
    println!("Downloaded: {}", paths.final_path.display());
    Ok(())
}

fn reconcile_partial_state(
    paths: &DownloadPaths,
    remote: &RemoteIdentity,
    metadata: Option<ResumeMetadata>,
) -> Result<Option<ResumeMetadata>> {
    let part_exists = paths.part_path.exists();
    match (part_exists, metadata) {
        (true, Some(metadata)) if metadata.matches_remote(remote) => Ok(Some(metadata)),
        (true, Some(_)) => {
            remove_file_if_exists(&paths.part_path)?;
            remove_file_if_exists(&paths.metadata_path)?;
            Ok(None)
        }
        (true, None) => {
            remove_file_if_exists(&paths.part_path)?;
            Ok(None)
        }
        (false, Some(_)) => {
            remove_file_if_exists(&paths.metadata_path)?;
            Ok(None)
        }
        (false, None) => Ok(None),
    }
}

fn stream_fresh_or_resumed(
    client: &Client,
    paths: &DownloadPaths,
    remote: &RemoteIdentity,
) -> Result<Option<String>> {
    let progress = download_progress(remote.expected_size, 0)?;
    let mut verified_hash = None;

    loop {
        let mut offset = file_len_if_exists(&paths.part_path)?;
        if offset > remote.expected_size {
            remove_file_if_exists(&paths.part_path)?;
            offset = 0;
        }
        progress.set_position(offset);

        if offset == remote.expected_size {
            progress.finish_and_clear();
            return Ok(verified_hash);
        }
        if offset > 0 {
            println!("Resuming at {}", HumanBytes(offset));
        }

        let mut request = client
            .request(Method::GET, &remote.url)
            .header(ACCEPT_ENCODING, "identity");
        if offset > 0 {
            request = request.header(RANGE, format!("bytes={offset}-"));
        }

        let mut response = request
            .send()
            .with_context(|| format!("Failed to download {}", remote.url))?;
        let status = response.status();

        if status == StatusCode::RANGE_NOT_SATISFIABLE {
            if offset == remote.expected_size {
                progress.finish_and_clear();
                return Ok(verified_hash);
            }
            remove_file_if_exists(&paths.part_path)?;
            progress.set_position(0);
            continue;
        }

        if status == StatusCode::OK && offset > 0 {
            remove_file_if_exists(&paths.part_path)?;
            progress.set_position(0);
            continue;
        }

        if status != StatusCode::OK && status != StatusCode::PARTIAL_CONTENT {
            bail!("Download failed with HTTP {status} for {}", remote.url);
        }

        if status == StatusCode::PARTIAL_CONTENT {
            validate_content_range(response.headers(), offset, remote.expected_size)?;
        }
        if offset > 0 || status == StatusCode::PARTIAL_CONTENT {
            validate_identity_encoding(&response)?;
        }

        let expected_hash = if offset == 0 {
            remote.strong_sha256()
        } else {
            None
        };
        let mut hasher = expected_hash.as_ref().map(|_| Sha256::new());
        let mut file = if offset == 0 {
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&paths.part_path)
        } else {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&paths.part_path)
        }
        .with_context(|| format!("Failed to open partial file {}", paths.part_path.display()))?;

        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let read = response.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            file.write_all(&buffer[..read])?;
            if let Some(hasher) = &mut hasher {
                hasher.update(&buffer[..read]);
            }
            progress.inc(read as u64);
        }
        file.flush()?;
        drop(file);

        let actual_size = file_len_if_exists(&paths.part_path)?;
        if actual_size != remote.expected_size {
            bail!(
                "Download ended at {} but expected {}; run tug again to resume",
                HumanBytes(actual_size),
                HumanBytes(remote.expected_size)
            );
        }

        if let (Some(expected_hash), Some(hasher)) = (expected_hash, hasher) {
            let actual_hash = hex_lower(&hasher.finalize());
            if actual_hash != expected_hash {
                fail_hash_mismatch(paths, remote, &expected_hash, &actual_hash)?;
            }
            verified_hash = Some(actual_hash);
        }
    }
}

fn finalize_download(
    paths: &DownloadPaths,
    remote: &RemoteIdentity,
    verified_hash: Option<String>,
) -> Result<()> {
    let actual_size = file_len_if_exists(&paths.part_path)?;
    if actual_size != remote.expected_size {
        bail!(
            "Refusing to finalize {}; expected {} but found {}",
            paths.part_path.display(),
            HumanBytes(remote.expected_size),
            HumanBytes(actual_size)
        );
    }

    let part_file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&paths.part_path)
        .with_context(|| format!("Failed to open partial file {}", paths.part_path.display()))?;
    part_file
        .sync_all()
        .with_context(|| format!("Failed to sync partial file {}", paths.part_path.display()))?;
    drop(part_file);

    let verified_hash = verify_completed_part_hash(paths, remote, verified_hash)?;

    #[cfg(windows)]
    if paths.final_path.exists() {
        fs::remove_file(&paths.final_path)?;
    }

    fs::rename(&paths.part_path, &paths.final_path).with_context(|| {
        format!(
            "Failed to finalize download from {} to {}",
            paths.part_path.display(),
            paths.final_path.display()
        )
    })?;
    let existing = read_resume_metadata(&paths.metadata_path).ok().flatten();
    write_completed_metadata(paths, remote, verified_hash, existing.as_ref())?;
    remove_file_if_exists(&paths.metadata_tmp_path)?;
    Ok(())
}

fn final_matches_remote(
    paths: &DownloadPaths,
    remote: &RemoteIdentity,
    metadata: Option<&ResumeMetadata>,
) -> Result<ExistingFinalMatch> {
    let stat = fs::metadata(&paths.final_path)
        .with_context(|| format!("Failed to inspect {}", paths.final_path.display()))?;
    if !stat.is_file() {
        bail!("{} exists but is not a file", paths.final_path.display());
    }
    if stat.len() != remote.expected_size {
        return Ok(ExistingFinalMatch::no());
    }
    let expected_hash = remote.strong_sha256();
    if let Some(completed) = metadata
        .filter(|m| m.matches_remote(remote))
        .and_then(|m| m.completed.as_ref())
    {
        if completed.size == stat.len() && completed.modified_at == modified_secs(&paths.final_path)
        {
            if let Some(expected_hash) = &expected_hash {
                if completed.sha256.as_deref() == Some(expected_hash.as_str()) {
                    return Ok(ExistingFinalMatch::yes(Some(expected_hash.clone())));
                }
            } else {
                return Ok(ExistingFinalMatch::yes(None));
            }
        }
    }
    if let Some(expected_hash) = expected_hash {
        let actual_hash = sha256_file(&paths.final_path)?;
        return Ok(if actual_hash == expected_hash {
            ExistingFinalMatch::yes(Some(actual_hash))
        } else {
            ExistingFinalMatch::no()
        });
    }
    Ok(ExistingFinalMatch::yes(None))
}

fn verify_completed_part_hash(
    paths: &DownloadPaths,
    remote: &RemoteIdentity,
    verified_hash: Option<String>,
) -> Result<Option<String>> {
    if verified_hash.is_some() {
        return Ok(verified_hash);
    }
    let Some(expected_hash) = remote.strong_sha256() else {
        return Ok(None);
    };

    let actual_hash = sha256_file(&paths.part_path)?;
    if actual_hash != expected_hash {
        fail_hash_mismatch(paths, remote, &expected_hash, &actual_hash)?;
    }
    Ok(Some(actual_hash))
}

fn fail_hash_mismatch(
    paths: &DownloadPaths,
    remote: &RemoteIdentity,
    expected_sha256: &str,
    actual_sha256: &str,
) -> Result<()> {
    quarantine_hash_mismatch(paths, remote, expected_sha256, actual_sha256)?;
    bail!(
        "Downloaded bytes did not match the expected SHA-256; final file was not written. Kept the failed payload at {} and wrote details to {}. This file is not resumable and is safe to delete.",
        paths.failed_part_path.display(),
        paths.failed_metadata_path.display()
    )
}

fn quarantine_hash_mismatch(
    paths: &DownloadPaths,
    remote: &RemoteIdentity,
    expected_sha256: &str,
    actual_sha256: &str,
) -> Result<()> {
    remove_file_if_exists(&paths.failed_metadata_tmp_path)?;
    remove_file_if_exists(&paths.failed_metadata_path)?;
    remove_file_if_exists(&paths.failed_part_path)?;

    fs::rename(&paths.part_path, &paths.failed_part_path).with_context(|| {
        format!(
            "Failed to quarantine hash-mismatched payload from {} to {}",
            paths.part_path.display(),
            paths.failed_part_path.display()
        )
    })?;
    remove_file_if_exists(&paths.metadata_path)?;
    remove_file_if_exists(&paths.metadata_tmp_path)?;

    let metadata = FailedHashMetadata {
        schema_version: RESUME_SCHEMA_VERSION,
        remote: remote.clone(),
        local_path: paths.final_path.to_string_lossy().to_string(),
        failed_payload_path: paths.failed_part_path.to_string_lossy().to_string(),
        expected_sha256: expected_sha256.to_string(),
        actual_sha256: actual_sha256.to_string(),
        size: fs::metadata(&paths.failed_part_path)?.len(),
        failed_at: now_secs(),
        reason: "sha256-mismatch".to_string(),
    };
    write_failed_hash_metadata_atomic(paths, &metadata)
}

fn write_failed_hash_metadata_atomic(
    paths: &DownloadPaths,
    metadata: &FailedHashMetadata,
) -> Result<()> {
    fs::create_dir_all(&paths.state_dir)?;
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&paths.failed_metadata_tmp_path)
        .with_context(|| {
            format!(
                "Failed to write temporary failure metadata {}",
                paths.failed_metadata_tmp_path.display()
            )
        })?;
    serde_json::to_writer_pretty(&mut file, metadata)?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    fs::rename(&paths.failed_metadata_tmp_path, &paths.failed_metadata_path).with_context(
        || {
            format!(
                "Failed to install failure metadata {}",
                paths.failed_metadata_path.display()
            )
        },
    )?;
    Ok(())
}

fn write_completed_metadata(
    paths: &DownloadPaths,
    remote: &RemoteIdentity,
    sha256: Option<String>,
    existing: Option<&ResumeMetadata>,
) -> Result<()> {
    let created_at = existing.map(|m| m.created_at).unwrap_or_else(now_secs);
    let completed = CompletedMetadata {
        size: fs::metadata(&paths.final_path)?.len(),
        modified_at: modified_secs(&paths.final_path),
        sha256,
    };
    write_resume_metadata_atomic(
        paths,
        &ResumeMetadata::new(remote.clone(), paths, created_at, Some(completed)),
    )
}

fn read_resume_metadata(path: &Path) -> Result<Option<ResumeMetadata>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path)
        .with_context(|| format!("Failed to read metadata {}", path.display()))?;
    Ok(Some(serde_json::from_str(&text).with_context(|| {
        format!("Failed to parse metadata {}", path.display())
    })?))
}

fn write_resume_metadata_atomic(paths: &DownloadPaths, metadata: &ResumeMetadata) -> Result<()> {
    fs::create_dir_all(&paths.state_dir)?;
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&paths.metadata_tmp_path)
        .with_context(|| {
            format!(
                "Failed to write temporary metadata {}",
                paths.metadata_tmp_path.display()
            )
        })?;
    serde_json::to_writer_pretty(&mut file, metadata)?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    fs::rename(&paths.metadata_tmp_path, &paths.metadata_path).with_context(|| {
        format!(
            "Failed to install metadata {}",
            paths.metadata_path.display()
        )
    })?;
    Ok(())
}

fn download_progress(total: u64, initial: u64) -> Result<ProgressBar> {
    let progress = ProgressBar::new(total);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
            .progress_chars("#>-"),
    );
    progress.set_position(initial);
    Ok(progress)
}

fn remote_size_from_headers(headers: &HeaderMap, status: StatusCode) -> Option<u64> {
    header_u64(headers, "X-Linked-Size")
        .or_else(|| content_range_total(headers))
        .or_else(|| {
            status
                .is_success()
                .then(|| header_u64(headers, "Content-Length"))
                .flatten()
        })
}

fn validate_content_range(
    headers: &HeaderMap,
    expected_start: u64,
    expected_total: u64,
) -> Result<()> {
    let content_range = header_value(headers, "Content-Range")
        .ok_or_else(|| anyhow!("Ranged response did not include Content-Range"))?;
    let (start, total) = parse_content_range_start_total(&content_range)
        .ok_or_else(|| anyhow!("Invalid Content-Range header: {content_range}"))?;
    if start != expected_start || total != expected_total {
        bail!(
            "Unexpected Content-Range {content_range}; expected start {expected_start} and total {expected_total}"
        );
    }
    Ok(())
}

fn validate_identity_encoding(response: &Response) -> Result<()> {
    if let Some(encoding) = response
        .headers()
        .get(CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
    {
        if !encoding.eq_ignore_ascii_case("identity") {
            bail!("Ranged response used unsupported Content-Encoding: {encoding}");
        }
    }
    Ok(())
}

fn content_range_total(headers: &HeaderMap) -> Option<u64> {
    header_value(headers, "Content-Range").and_then(|v| parse_content_range_total(&v))
}

fn parse_content_range_total(value: &str) -> Option<u64> {
    value.rsplit_once('/')?.1.parse().ok()
}

fn parse_content_range_start_total(value: &str) -> Option<(u64, u64)> {
    let value = value.strip_prefix("bytes ")?;
    let (range, total) = value.split_once('/')?;
    let (start, _) = range.split_once('-')?;
    Some((start.parse().ok()?, total.parse().ok()?))
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
    header_value(headers, name)?.parse().ok()
}

fn file_len_if_exists(path: &Path) -> Result<u64> {
    match fs::metadata(path) {
        Ok(metadata) => {
            if !metadata.is_file() {
                bail!("{} exists but is not a file", path.display());
            }
            Ok(metadata.len())
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(err) => Err(err).with_context(|| format!("Failed to inspect {}", path.display())),
    }
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("Failed to remove {}", path.display())),
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn modified_secs(path: &Path) -> Option<u64> {
    fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
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
    #[serde(default, rename = "createdAt")]
    created_at: Option<String>,
    #[serde(default, rename = "lastModified")]
    last_modified: Option<String>,
}

/// Processed file info for display.
pub struct FileInfo {
    path: String,
    size: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
struct RemotePath {
    original: String,
    components: Vec<String>,
}

impl RemotePath {
    fn parse(input: &str) -> Result<Self> {
        if input.is_empty() {
            bail!("Model path cannot be empty");
        }
        if input.starts_with('/') || input.starts_with('\\') || input.contains('\\') {
            bail!("Model path must be a repo-relative path: {input}");
        }

        let components: Vec<String> = input
            .split('/')
            .map(|component| {
                if component.is_empty() || component == "." || component == ".." {
                    bail!("Model path contains an unsafe component: {input}");
                }
                Ok(component.to_string())
            })
            .collect::<Result<_>>()?;

        Ok(Self {
            original: input.to_string(),
            components,
        })
    }

    fn as_str(&self) -> &str {
        &self.original
    }

    fn join_under(&self, base: &Path) -> PathBuf {
        self.components
            .iter()
            .fold(base.to_path_buf(), |path, component| path.join(component))
    }
}

#[derive(Clone, Debug)]
struct DownloadPaths {
    final_path: PathBuf,
    state_dir: PathBuf,
    part_path: PathBuf,
    metadata_path: PathBuf,
    metadata_tmp_path: PathBuf,
    failed_part_path: PathBuf,
    failed_metadata_path: PathBuf,
    failed_metadata_tmp_path: PathBuf,
    lock_path: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RemoteIdentity {
    repo_id: String,
    path: String,
    revision: String,
    resolved_commit: String,
    url: String,
    expected_size: u64,
    etag: Option<String>,
    linked_etag: Option<String>,
    xet_hash: Option<String>,
    accept_ranges: bool,
}

impl RemoteIdentity {
    fn same_file(&self, other: &RemoteIdentity) -> bool {
        self.repo_id == other.repo_id
            && self.path == other.path
            && self.revision == other.revision
            && self.resolved_commit == other.resolved_commit
            && self.expected_size == other.expected_size
            && self.etag == other.etag
            && self.linked_etag == other.linked_etag
            && self.xet_hash == other.xet_hash
    }

    fn strong_sha256(&self) -> Option<String> {
        self.linked_etag
            .as_deref()
            .or(self.etag.as_deref())
            .and_then(normalize_sha256)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ResumeMetadata {
    schema_version: u8,
    remote: RemoteIdentity,
    local_path: String,
    created_at: u64,
    last_seen_at: u64,
    completed: Option<CompletedMetadata>,
}

impl ResumeMetadata {
    fn new(
        remote: RemoteIdentity,
        paths: &DownloadPaths,
        created_at: u64,
        completed: Option<CompletedMetadata>,
    ) -> Self {
        Self {
            schema_version: RESUME_SCHEMA_VERSION,
            remote,
            local_path: paths.final_path.to_string_lossy().to_string(),
            created_at,
            last_seen_at: now_secs(),
            completed,
        }
    }

    fn matches_remote(&self, remote: &RemoteIdentity) -> bool {
        self.schema_version == RESUME_SCHEMA_VERSION && self.remote.same_file(remote)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompletedMetadata {
    size: u64,
    modified_at: Option<u64>,
    sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FailedHashMetadata {
    schema_version: u8,
    remote: RemoteIdentity,
    local_path: String,
    failed_payload_path: String,
    expected_sha256: String,
    actual_sha256: String,
    size: u64,
    failed_at: u64,
    reason: String,
}

#[derive(Clone, Debug)]
struct ExistingFinalMatch {
    matches: bool,
    sha256: Option<String>,
}

impl ExistingFinalMatch {
    fn yes(sha256: Option<String>) -> Self {
        Self {
            matches: true,
            sha256,
        }
    }

    fn no() -> Self {
        Self {
            matches: false,
            sha256: None,
        }
    }
}

fn normalize_sha256(value: &str) -> Option<String> {
    let normalized = value.trim().trim_start_matches("W/").trim_matches('"');
    if normalized.len() == 64 && normalized.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(normalized.to_ascii_lowercase())
    } else {
        None
    }
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

    #[test]
    fn test_resolve_url_encodes_path_segments() {
        let repo: RepoId = "org/repo".parse().unwrap();
        assert_eq!(
            resolve_url(&repo, "nested/model name.gguf"),
            "https://huggingface.co/org/repo/resolve/main/nested/model%20name.gguf"
        );
    }

    #[test]
    fn test_remote_path_rejects_traversal() {
        assert!(RemotePath::parse("../model.gguf").is_err());
        assert!(RemotePath::parse("a/../model.gguf").is_err());
        assert!(RemotePath::parse("/model.gguf").is_err());
        assert!(RemotePath::parse("a//model.gguf").is_err());
        assert!(RemotePath::parse("a\\model.gguf").is_err());
    }

    #[test]
    fn test_download_paths_for_nested_target() {
        let tmp = tempfile::tempdir().unwrap();
        let remote_path = RemotePath::parse("Q2_K_L/foo.gguf").unwrap();
        let paths = download_paths(tmp.path(), &remote_path).unwrap();
        assert_eq!(paths.final_path, tmp.path().join("Q2_K_L/foo.gguf"));
        assert_eq!(paths.state_dir, tmp.path().join("Q2_K_L/.hugtug"));
        assert_eq!(
            paths.part_path,
            tmp.path().join("Q2_K_L/.hugtug/foo.gguf.part")
        );
        assert_eq!(
            paths.metadata_path,
            tmp.path().join("Q2_K_L/.hugtug/foo.gguf.json")
        );
        assert_eq!(
            paths.failed_part_path,
            tmp.path()
                .join("Q2_K_L/.hugtug/foo.gguf.failed-sha256.part")
        );
        assert_eq!(
            paths.failed_metadata_path,
            tmp.path()
                .join("Q2_K_L/.hugtug/foo.gguf.failed-sha256.json")
        );
        assert_eq!(
            paths.lock_path,
            tmp.path().join("Q2_K_L/.hugtug/foo.gguf.lock")
        );
    }

    #[test]
    fn test_hash_mismatch_quarantines_part_and_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let remote_path = RemotePath::parse("Q2_K_L/foo.gguf").unwrap();
        let paths = download_paths(tmp.path(), &remote_path).unwrap();
        ensure_download_dirs(&paths).unwrap();
        std::fs::write(&paths.part_path, b"bad").unwrap();
        std::fs::write(&paths.metadata_path, b"{}").unwrap();

        let remote = RemoteIdentity {
            repo_id: "org/repo".to_string(),
            path: remote_path.as_str().to_string(),
            revision: DEFAULT_REVISION.to_string(),
            resolved_commit: "0123456789012345678901234567890123456789".to_string(),
            url: "https://huggingface.co/org/repo/resolve/0123456789012345678901234567890123456789/Q2_K_L/foo.gguf".to_string(),
            expected_size: 3,
            etag: None,
            linked_etag: None,
            xet_hash: None,
            accept_ranges: true,
        };

        quarantine_hash_mismatch(&paths, &remote, "expected", "actual").unwrap();

        assert!(!paths.part_path.exists());
        assert!(!paths.metadata_path.exists());
        assert_eq!(std::fs::read(&paths.failed_part_path).unwrap(), b"bad");

        let text = std::fs::read_to_string(&paths.failed_metadata_path).unwrap();
        let metadata: FailedHashMetadata = serde_json::from_str(&text).unwrap();
        assert_eq!(metadata.reason, "sha256-mismatch");
        assert_eq!(metadata.expected_sha256, "expected");
        assert_eq!(metadata.actual_sha256, "actual");
        assert_eq!(metadata.size, 3);
        assert_eq!(metadata.remote.path, "Q2_K_L/foo.gguf");
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

    #[test]
    fn test_parse_content_range_start_and_total() {
        assert_eq!(
            parse_content_range_start_total("bytes 123-456/789"),
            Some((123, 789))
        );
        assert_eq!(parse_content_range_start_total("invalid"), None);
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
