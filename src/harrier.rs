use std::fmt;

pub const DIM: usize = 640;
const BOS: u32 = 2;
const EOS: u32 = 1;
/// Cap for one text: workspace holds a T x T score matrix; 4096 is plenty for
/// agent memories. Longer texts are truncated.
pub const MAX_TOKENS: usize = 4096;

mod ffi {
    #[link(name = "harrier")]
    unsafe extern "C" {
        pub fn hr_workspace_bytes(max_tokens: u32) -> usize;
        pub fn hr_embed(
            workspace: *mut u8,
            workspace_size: usize,
            ids: *const i32,
            n_tokens: u32,
            out: *mut f32,
        ) -> i32;
        pub fn hr_tokenize(
            utf8: *const u8,
            len: usize,
            ids: *mut u32,
            ids_cap: usize,
            scratch: *mut u32,
            scratch_cap: usize,
        ) -> usize;
    }
}

#[derive(Debug)]
pub enum Error {
    BadWeights,
    BadArgs,
    NoSpace,
    Tokenize,
    Unknown(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::BadWeights => write!(f, "baked weights failed validation (build issue)"),
            Error::BadArgs => write!(f, "bad arguments (empty text / token out of range)"),
            Error::NoSpace => write!(f, "workspace too small"),
            Error::Tokenize => write!(f, "tokenizer failure"),
            Error::Unknown(c) => write!(f, "harrier unknown error {c}"),
        }
    }
}
impl std::error::Error for Error {}

pub struct Harrier;

unsafe impl Send for Harrier {}

impl Default for Harrier {
    fn default() -> Self {
        Harrier
    }
}
unsafe impl Sync for Harrier {} // weights/tokenizer live in .rodata; workspace is per-call

impl Harrier {
    /// Everything is baked in at build time — no arguments, no runtime assets.
    pub fn new() -> Result<Self, Error> {
        // force a cheap validation pass through the C side on first use is
        // unnecessary: hr_embed/hr_tokenize validate per call.
        Ok(Self)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i32>, Error> {
        let bytes = text.as_bytes();
        let scratch_cap = 6 * bytes.len() + 8;
        let mut scratch = vec![0u32; scratch_cap];
        let need = unsafe {
            ffi::hr_tokenize(
                bytes.as_ptr(),
                bytes.len(),
                std::ptr::null_mut(),
                0,
                scratch.as_mut_ptr(),
                scratch_cap,
            )
        };
        if need == usize::MAX {
            return Err(Error::Tokenize);
        }
        let mut ids32 = vec![0u32; need];
        let n = unsafe {
            ffi::hr_tokenize(
                bytes.as_ptr(),
                bytes.len(),
                ids32.as_mut_ptr(),
                need,
                scratch.as_mut_ptr(),
                scratch_cap,
            )
        };
        if n == usize::MAX || n > need {
            return Err(Error::Tokenize);
        }
        let mut ids: Vec<i32> = Vec::with_capacity(n.min(MAX_TOKENS));
        for &id in ids32.iter().take(n) {
            if ids.len() == MAX_TOKENS - 1 {
                break;
            }
            ids.push(id as i32);
        }
        if ids.first() != Some(&(BOS as i32)) {
            ids.insert(0, BOS as i32);
        }
        if ids.last() != Some(&(EOS as i32)) {
            ids.push(EOS as i32);
        }
        Ok(ids)
    }

    fn forward(&self, ids: &[i32]) -> Result<[f32; DIM], Error> {
        let ws_size = unsafe { ffi::hr_workspace_bytes(ids.len() as u32) };
        let mut ws = vec![0u8; ws_size];
        let mut out = [0f32; DIM];
        let rc = unsafe {
            ffi::hr_embed(
                ws.as_mut_ptr(),
                ws.len(),
                ids.as_ptr(),
                ids.len() as u32,
                out.as_mut_ptr(),
            )
        };
        match rc {
            0 => Ok(out),
            -1 => Err(Error::BadArgs),
            -2 => Err(Error::BadWeights),
            -3 => Err(Error::NoSpace),
            c => Err(Error::Unknown(c)),
        }
    }

    /// Embed texts (document side; no instruction prefix). L2-normalized.
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<[f32; DIM]>, Error> {
        texts
            .iter()
            .map(|t| self.forward(&self.encode(t)?))
            .collect()
    }

    /// Parallel `embed`: one OS thread per chunk, each with its own workspace.
    pub fn embed_par(&self, texts: &[&str]) -> Result<Vec<[f32; DIM]>, Error> {
        if texts.len() <= 1 {
            return self.embed(texts);
        }
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(texts.len());
        let chunk = texts.len().div_ceil(n_threads);
        let mut out: Vec<Option<[f32; DIM]>> = vec![None; texts.len()];
        let mut errs: Vec<Option<String>> = vec![None; texts.len()];
        std::thread::scope(|s| {
            for ((ts, os), es) in texts
                .chunks(chunk)
                .zip(out.chunks_mut(chunk))
                .zip(errs.chunks_mut(chunk))
            {
                s.spawn(move || {
                    for (i, t) in ts.iter().enumerate() {
                        match self.encode(t).and_then(|ids| self.forward(&ids)) {
                            Ok(e) => os[i] = Some(e),
                            Err(err) => es[i] = Some(err.to_string()),
                        }
                    }
                });
            }
        });
        if errs.iter().any(|e| e.is_some()) {
            return Err(Error::Tokenize);
        }
        Ok(out.into_iter().map(|o| o.unwrap()).collect())
    }

    /// Embed one query with the harrier web-search instruction prefix.
    pub fn embed_query(&self, query: &str) -> Result<[f32; DIM], Error> {
        let q = format!(
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}"
        );
        self.forward(&self.encode(&q)?)
    }

    /// Rerank docs against a query: cosine scores in doc order (not sorted).
    pub fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>, Error> {
        let q = self.embed_query(query)?;
        let ds = self.embed(docs)?;
        Ok(ds
            .iter()
            .map(|d| q.iter().zip(d).map(|(a, b)| a * b).sum())
            .collect())
    }
}

impl Harrier {
    /// Token count for a text under the baked-in tokenizer (dev helper).
    pub fn token_count(&self, text: &str) -> usize {
        self.encode(text).map(|v| v.len()).unwrap_or(0)
    }
}
