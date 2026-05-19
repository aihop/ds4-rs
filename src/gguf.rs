use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::MmapOptions;

use crate::error::{Ds4Error, Result};

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_MAX_DIMS: usize = 4;

#[derive(Clone, Debug, Default)]
pub struct GgufModel {
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
    pub alignment: u32,
    pub file_size: u64,
    pub tensor_data_pos: u64,
    pub architecture: Option<String>,
    pub vocab_size: Option<u32>,
    pub tokenizer_tokens: Vec<String>,
    pub tokenizer_merges: Vec<String>,
    pub tensors: Vec<GgufTensor>,
    pub tensors_by_name: HashMap<String, usize>,
    pub(crate) data: Arc<GgufData>,
}

impl GgufModel {
    pub fn model_map_ptr(&self) -> *const std::ffi::c_void {
        match &*self.data {
            GgufData::Mapped(mmap) => mmap.as_ptr() as *const std::ffi::c_void,
            GgufData::Owned(vec) => vec.as_ptr() as *const std::ffi::c_void,
            GgufData::Empty => std::ptr::null(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum GgufData {
    Empty,
    #[allow(dead_code)]
    Owned(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl Default for GgufData {
    fn default() -> Self {
        Self::Empty
    }
}

impl GgufData {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Empty => &[],
            Self::Owned(bytes) => bytes.as_slice(),
            Self::Mapped(map) => map.as_ref(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GgufTensor {
    pub name: String,
    pub ndim: u32,
    pub dims: Vec<u64>,
    pub tensor_type: u32,
    pub rel_offset: u64,
    pub abs_offset: u64,
    pub elements: u64,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(raw: u32) -> Result<Self> {
        let ty = match raw {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => {
                return Err(Ds4Error::Protocol(format!(
                    "unsupported GGUF metadata type: {raw}"
                )))
            }
        };
        Ok(ty)
    }

    fn scalar_size(self) -> Option<u64> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

pub fn load_model(path: &Path) -> Result<GgufModel> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    if file_size < 32 {
        return Err(Ds4Error::Protocol(format!(
            "model file is too small to be GGUF: {}",
            path.display()
        )));
    }
    // SAFETY: The file is mapped into memory. We must ensure that the underlying file is not modified
    // concurrently by another process, as this could cause undefined behavior. We treat the GGUF file as read-only.
    let mapped = unsafe { MmapOptions::new().map(&file)? };
    let mut reader = Reader::new(mapped.as_ref());

    let magic = reader.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(Ds4Error::Protocol(format!(
            "model is not a GGUF file: {}",
            path.display()
        )));
    }

    let version = reader.read_u32()?;
    if version != 3 {
        return Err(Ds4Error::Protocol(format!(
            "only GGUF v3 is supported, got v{version}"
        )));
    }

    let n_tensors = reader.read_u64()?;
    let n_kv = reader.read_u64()?;

    let mut model = GgufModel {
        version,
        n_tensors,
        n_kv,
        alignment: 32,
        file_size,
        tensor_data_pos: 0,
        architecture: None,
        vocab_size: None,
        tokenizer_tokens: Vec::new(),
        tokenizer_merges: Vec::new(),
        tensors: Vec::new(),
        tensors_by_name: HashMap::new(),
        data: Arc::new(GgufData::Empty),
    };

    for _ in 0..n_kv {
        let key = reader.read_string()?;
        let value_type = GgufValueType::from_u32(reader.read_u32()?)?;
        match key.as_str() {
            "general.alignment" if value_type == GgufValueType::Uint32 => {
                model.alignment = reader.read_u32()?;
            }
            "general.architecture" if value_type == GgufValueType::String => {
                model.architecture = Some(reader.read_string()?);
            }
            "deepseek4.vocab_size" if matches!(value_type, GgufValueType::Uint32 | GgufValueType::Uint64) => {
                model.vocab_size = Some(match value_type {
                    GgufValueType::Uint32 => reader.read_u32()?,
                    GgufValueType::Uint64 => reader.read_u64()? as u32,
                    _ => unreachable!(),
                });
            }
            "tokenizer.ggml.tokens" if value_type == GgufValueType::Array => {
                let (item_type, len) = read_array_header(&mut reader)?;
                if item_type != GgufValueType::String {
                    return Err(Ds4Error::Protocol(
                        "GGUF tokenizer token table is missing or invalid".to_string(),
                    ));
                }
                model.tokenizer_tokens = read_string_array(&mut reader, len)?;
            }
            "tokenizer.ggml.merges" if value_type == GgufValueType::Array => {
                let (item_type, len) = read_array_header(&mut reader)?;
                if item_type != GgufValueType::String {
                    return Err(Ds4Error::Protocol(
                        "GGUF tokenizer merge table is missing or invalid".to_string(),
                    ));
                }
                model.tokenizer_merges = read_string_array(&mut reader, len)?;
            }
            _ => skip_value(&mut reader, value_type, 0)?,
        }
    }

    parse_tensors(&mut reader, &mut model)?;
    model.data = Arc::new(GgufData::Mapped(mapped));

    Ok(model)
}

impl GgufModel {
    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors_by_name
            .get(name)
            .and_then(|idx| self.tensors.get(*idx))
    }

    pub fn tensor_bytes(&self, tensor: &GgufTensor) -> Option<&[u8]> {
        let start = usize::try_from(tensor.abs_offset).ok()?;
        let len = usize::try_from(tensor.bytes).ok()?;
        let end = start.checked_add(len)?;
        self.data.as_slice().get(start..end)
    }
}

fn parse_tensors(reader: &mut Reader, model: &mut GgufModel) -> Result<()> {
    let mut tensors = Vec::with_capacity(model.n_tensors.min(4096) as usize);
    let mut tensors_by_name = HashMap::with_capacity(model.n_tensors.min(4096) as usize);

    for _ in 0..model.n_tensors {
        let name = reader.read_string()?;
        let ndim = reader.read_u32()?;
        if ndim == 0 || ndim as usize > GGUF_MAX_DIMS {
            return Err(Ds4Error::Protocol(
                "tensor has an unsupported number of dimensions".to_string(),
            ));
        }

        let mut dims = Vec::with_capacity(ndim as usize);
        let mut elements = 1u64;
        for _ in 0..ndim {
            let dim = reader.read_u64()?;
            if dim != 0 && elements > u64::MAX / dim {
                return Err(Ds4Error::Protocol(
                    "tensor element count overflow".to_string(),
                ));
            }
            elements = elements.saturating_mul(dim);
            dims.push(dim);
        }

        let tensor_type = reader.read_u32()?;
        let rel_offset = reader.read_u64()?;
        let bytes = tensor_nbytes(tensor_type, elements).unwrap_or(0);

        let idx = tensors.len();
        tensors_by_name.insert(name.clone(), idx);
        tensors.push(GgufTensor {
            name,
            ndim,
            dims,
            tensor_type,
            rel_offset,
            abs_offset: 0,
            elements,
            bytes,
        });
    }

    let tensor_data_pos = align_up(reader.pos(), model.alignment as u64);
    for tensor in &mut tensors {
        tensor.abs_offset = tensor_data_pos
            .checked_add(tensor.rel_offset)
            .ok_or_else(|| Ds4Error::Protocol("tensor offset overflow".to_string()))?;
        if tensor.bytes != 0
            && (tensor.abs_offset > model.file_size
                || tensor.bytes > model.file_size.saturating_sub(tensor.abs_offset))
        {
            return Err(Ds4Error::Protocol(
                "tensor points outside GGUF file".to_string(),
            ));
        }
    }

    model.tensor_data_pos = tensor_data_pos;
    model.tensors = tensors;
    model.tensors_by_name = tensors_by_name;
    Ok(())
}

fn tensor_nbytes(tensor_type: u32, elements: u64) -> Option<u64> {
    let (block_elems, block_bytes) = match tensor_type {
        0 => (1u64, 4u64),   // f32
        1 => (1, 2),         // f16
        2 => (32, 18),       // q4_0
        3 => (32, 20),       // q4_1
        6 => (32, 22),       // q5_0
        7 => (32, 24),       // q5_1
        8 => (32, 34),       // q8_0
        9 => (32, 40),       // q8_1
        10 => (256, 84),     // q2_k
        11 => (256, 110),    // q3_k
        12 => (256, 144),    // q4_k
        13 => (256, 176),    // q5_k
        14 => (256, 210),    // q6_k
        15 => (256, 292),    // q8_k
        16 => (256, 66),     // iq2_xxs
        17 => (256, 74),     // iq2_xs
        18 => (256, 98),     // iq3_xxs
        19 => (256, 110),    // iq1_s
        20 => (256, 50),     // iq4_nl
        21 => (256, 110),    // iq3_s
        22 => (256, 82),     // iq2_s
        23 => (256, 136),    // iq4_xs
        24 => (1, 1),        // i8
        25 => (1, 2),        // i16
        26 => (1, 4),        // i32
        27 => (1, 8),        // i64
        28 => (1, 8),        // f64
        29 => (256, 56),     // iq1_m
        30 => (1, 2),        // bf16
        _ => return None,
    };
    let blocks = elements.checked_add(block_elems - 1)? / block_elems;
    blocks.checked_mul(block_bytes)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + (alignment - rem)
    }
}

fn read_array_header(reader: &mut Reader) -> Result<(GgufValueType, u64)> {
    let item_type = GgufValueType::from_u32(reader.read_u32()?)?;
    let len = reader.read_u64()?;
    Ok((item_type, len))
}

fn read_string_array(reader: &mut Reader, len: u64) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(len.min(4096) as usize);
    for _ in 0..len {
        out.push(reader.read_string()?);
    }
    Ok(out)
}

fn skip_value(reader: &mut Reader, value_type: GgufValueType, depth: usize) -> Result<()> {
    if depth > 8 {
        return Err(Ds4Error::Protocol(
            "GGUF metadata array nesting is too deep".to_string(),
        ));
    }

    if let Some(size) = value_type.scalar_size() {
        reader.skip_bytes(size)?;
        return Ok(());
    }

    match value_type {
        GgufValueType::String => {
            let _ = reader.read_string()?;
            Ok(())
        }
        GgufValueType::Array => {
            let (item_type, len) = read_array_header(reader)?;
            if let Some(item_size) = item_type.scalar_size() {
                reader.skip_bytes(len.saturating_mul(item_size))
            } else {
                for _ in 0..len {
                    skip_value(reader, item_type, depth + 1)?;
                }
                Ok(())
            }
        }
        _ => Err(Ds4Error::Protocol(
            "unsupported GGUF metadata shape".to_string(),
        )),
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn pos(&self) -> u64 {
        self.pos as u64
    }

    fn skip_bytes(&mut self, bytes: u64) -> Result<()> {
        let end = self
            .pos
            .checked_add(bytes as usize)
            .ok_or_else(|| Ds4Error::Protocol("truncated GGUF file".to_string()))?;
        if end > self.bytes.len() {
            return Err(Ds4Error::Protocol("truncated GGUF file".to_string()));
        }
        self.pos = end;
        Ok(())
    }

    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .pos
            .checked_add(N)
            .ok_or_else(|| Ds4Error::Protocol("truncated GGUF file".to_string()))?;
        if end > self.bytes.len() {
            return Err(Ds4Error::Protocol("truncated GGUF file".to_string()));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..end]);
        self.pos = end;
        Ok(out)
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact()?))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact()?))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| Ds4Error::Protocol("truncated GGUF file".to_string()))?;
        if end > self.bytes.len() {
            return Err(Ds4Error::Protocol("truncated GGUF file".to_string()));
        }
        let bytes = self.bytes[self.pos..end].to_vec();
        self.pos = end;
        String::from_utf8(bytes).map_err(|_| {
            Ds4Error::Protocol("GGUF string metadata is not valid UTF-8".to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_tensor_directory_and_offsets() {
        let path = temp_path("gguf_tensor_parse");
        fs::write(&path, build_test_gguf()).unwrap();
        let model = load_model(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(model.version, 3);
        assert_eq!(model.n_tensors, 1);
        assert_eq!(model.alignment, 32);
        assert_eq!(model.tensor_data_pos, 192);
        let tensor = model.tensor("tok_embeddings.weight").unwrap();
        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.dims, vec![4, 8]);
        assert_eq!(tensor.tensor_type, 0);
        assert_eq!(tensor.elements, 32);
        assert_eq!(tensor.bytes, 128);
        assert_eq!(tensor.abs_offset, 192);
    }

    #[test]
    fn owned_backing_still_serves_tensor_bytes() {
        let bytes = build_test_gguf();
        let mut model = GgufModel {
            version: 3,
            n_tensors: 0,
            n_kv: 0,
            alignment: 32,
            file_size: bytes.len() as u64,
            tensor_data_pos: 0,
            architecture: None,
            vocab_size: None,
            tokenizer_tokens: Vec::new(),
            tokenizer_merges: Vec::new(),
            tensors: vec![GgufTensor {
                name: "tok_embeddings.weight".to_string(),
                ndim: 2,
                dims: vec![4, 8],
                tensor_type: 0,
                rel_offset: 0,
                abs_offset: 192,
                elements: 32,
                bytes: 128,
            }],
            tensors_by_name: HashMap::from([("tok_embeddings.weight".to_string(), 0usize)]),
            data: Arc::new(GgufData::Owned(bytes)),
        };
        let tensor = model.tensor("tok_embeddings.weight").unwrap().clone();
        assert_eq!(model.tensor_bytes(&tensor).unwrap().len(), 128);
        model.tensor_data_pos = 192;
    }

    fn temp_path(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}_{nanos}.gguf"))
    }

    fn build_test_gguf() -> Vec<u8> {
        let mut out = Vec::new();
        push_u32(&mut out, GGUF_MAGIC);
        push_u32(&mut out, 3);
        push_u64(&mut out, 1); // n_tensors
        push_u64(&mut out, 2); // n_kv

        push_string(&mut out, "general.alignment");
        push_u32(&mut out, GgufValueType::Uint32 as u32);
        push_u32(&mut out, 32);

        push_string(&mut out, "general.architecture");
        push_u32(&mut out, GgufValueType::String as u32);
        push_string(&mut out, "deepseek4");

        push_string(&mut out, "tok_embeddings.weight");
        push_u32(&mut out, 2);
        push_u64(&mut out, 4);
        push_u64(&mut out, 8);
        push_u32(&mut out, 0); // f32
        push_u64(&mut out, 0);

        let aligned = align_up(out.len() as u64, 32) as usize;
        out.resize(aligned, 0);
        out.resize(aligned + 128, 0);
        out
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(out: &mut Vec<u8>, value: &str) {
        push_u64(out, value.len() as u64);
        out.extend_from_slice(value.as_bytes());
    }
}
