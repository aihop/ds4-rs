use std::collections::HashMap;

use crate::engine::{Engine, ThinkMode};
use crate::error::{Ds4Error, Result};
use crate::gguf::GgufModel;
use crate::types::Tokens;

/// A text tokenizer mapping strings to sequences of token IDs.
#[derive(Clone, Debug, Default)]
pub struct Tokenizer {
    mode: TokenizerMode,
}

#[derive(Clone, Debug, Default)]
enum TokenizerMode {
    #[default]
    Preview,
    Gguf(Vocabulary),
}

#[derive(Clone, Debug)]
struct Vocabulary {
    tokens: Vec<String>,
    token_to_id: HashMap<String, i32>,
    merges: HashMap<String, i32>,
    bos_id: i32,
    eos_id: i32,
    user_id: i32,
    assistant_id: i32,
    think_start_id: i32,
    think_end_id: i32,
    dsml_id: i32,
}

impl Tokenizer {
    /// Creates a stub preview tokenizer for testing without a real model.
    pub fn preview() -> Self {
        Self {
            mode: TokenizerMode::Preview,
        }
    }

    /// Loads the tokenizer from a given `GgufModel`.
    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        if model.tokenizer_tokens.is_empty() {
            return Err(Ds4Error::Protocol(
                "GGUF tokenizer token table is missing or invalid".to_string(),
            ));
        }
        let mut token_to_id = HashMap::with_capacity(model.tokenizer_tokens.len());
        for (idx, token) in model.tokenizer_tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), idx as i32);
        }
        let mut merges = HashMap::with_capacity(model.tokenizer_merges.len());
        for (idx, merge) in model.tokenizer_merges.iter().enumerate() {
            merges.insert(merge.clone(), idx as i32);
        }
        let bos_id = required_token_id(&token_to_id, "<｜begin▁of▁sentence｜>")?;
        let eos_id = required_token_id(&token_to_id, "<｜end▁of▁sentence｜>")?;
        let user_id = required_token_id(&token_to_id, "<｜User｜>")?;
        let assistant_id = required_token_id(&token_to_id, "<｜Assistant｜>")?;
        let think_start_id = required_token_id(&token_to_id, "<think>")?;
        let think_end_id = required_token_id(&token_to_id, "</think>")?;
        let dsml_id = required_token_id(&token_to_id, "｜DSML｜")?;
        Ok(Self {
            mode: TokenizerMode::Gguf(Vocabulary {
                tokens: model.tokenizer_tokens.clone(),
                token_to_id,
                merges,
                bos_id,
                eos_id,
                user_id,
                assistant_id,
                think_start_id,
                think_end_id,
                dsml_id,
            }),
        })
    }

    pub fn tokenize_text(&self, text: &str) -> Tokens {
        match &self.mode {
            TokenizerMode::Preview => Tokens(text.bytes().map(i32::from).collect()),
            TokenizerMode::Gguf(vocab) => tokenize_text_bpe(vocab, text),
        }
    }

    pub fn tokenize_rendered_chat(&self, text: &str) -> Tokens {
        match &self.mode {
            TokenizerMode::Preview => Tokens(text.bytes().map(i32::from).collect()),
            TokenizerMode::Gguf(vocab) => tokenize_rendered_chat_vocab(vocab, text),
        }
    }

    pub fn token_text(&self, token: i32) -> String {
        match &self.mode {
            TokenizerMode::Preview => char::from_u32(token as u32).unwrap_or('?').to_string(),
            TokenizerMode::Gguf(vocab) => vocab
                .tokens
                .get(token as usize)
                .cloned()
                .unwrap_or_else(|| "?".to_string()),
        }
    }

    /// Decodes a list of token IDs into a contiguous string.
    pub fn decode_tokens(&self, tokens: &Tokens) -> String {
        match &self.mode {
            TokenizerMode::Preview => tokens
                .0
                .iter()
                .filter_map(|token| char::from_u32(*token as u32))
                .collect(),
            TokenizerMode::Gguf(vocab) => decode_tokens_vocab(vocab, tokens),
        }
    }

    /// Renders a chat conversation into token IDs, including system prompt.
    pub fn render_chat_prompt(&self, system: &str, prompt: &str, think_mode: ThinkMode) -> Tokens {
        match &self.mode {
            TokenizerMode::Preview => {
                let think = match think_mode {
                    ThinkMode::None => "</think>",
                    ThinkMode::High => "<think>",
                    ThinkMode::Max => "<think max>",
                };
                let rendered = format!(
                    "<｜System｜>{system}\n<｜User｜>{prompt}\n<｜Assistant｜>{think}\n"
                );
                Tokens(rendered.bytes().map(i32::from).collect())
            }
            TokenizerMode::Gguf(vocab) => {
                let mut out = Tokens::default();
                out.push(vocab.bos_id);
                if matches!(think_mode, ThinkMode::Max) {
                    append_text(vocab, "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\n", &mut out);
                }
                if !system.is_empty() {
                    append_text(vocab, system, &mut out);
                }
                out.push(vocab.user_id);
                append_text(vocab, prompt, &mut out);
                out.push(vocab.assistant_id);
                out.push(if matches!(think_mode, ThinkMode::None) {
                    vocab.think_end_id
                } else {
                    vocab.think_start_id
                });
                out
            }
        }
    }

    pub fn is_real(&self) -> bool {
        matches!(self.mode, TokenizerMode::Gguf(_))
    }

    pub fn vocab_size(&self) -> Option<usize> {
        match &self.mode {
            TokenizerMode::Preview => None,
            TokenizerMode::Gguf(vocab) => Some(vocab.tokens.len()),
        }
    }
}

pub fn render_chat_prompt(engine: &Engine, system: &str, prompt: &str, think_mode: ThinkMode) -> Tokens {
    engine.render_chat_prompt(system, prompt, think_mode)
}

fn append_text(vocab: &Vocabulary, text: &str, out: &mut Tokens) {
    let encoded = tokenize_text_bpe(vocab, text);
    out.0.extend(encoded.0);
}

fn tokenize_text_bpe(vocab: &Vocabulary, text: &str) -> Tokens {
    if text.is_empty() {
        return Tokens::default();
    }
    if let Some(token) = vocab.token_to_id.get(text) {
        return Tokens(vec![*token]);
    }

    let mut out = Tokens::default();
    for (start, end) in joyai_pieces(text) {
        bpe_emit_piece(vocab, &text[start..end], &mut out);
    }
    out
}

fn tokenize_rendered_chat_vocab(vocab: &Vocabulary, text: &str) -> Tokens {
    if text.is_empty() {
        return Tokens::default();
    }

    let mut out = Tokens::default();
    let mut span_start = 0usize;
    let mut pos = 0usize;
    while pos < text.len() {
        if let Some((token, len)) = special_token_at(vocab, &text[pos..]) {
            tokenize_span(vocab, &text[span_start..pos], &mut out);
            out.push(token);
            pos += len;
            span_start = pos;
            continue;
        }
        pos = next_utf8_char(text, pos);
    }
    tokenize_span(vocab, &text[span_start..], &mut out);
    out
}

fn tokenize_span(vocab: &Vocabulary, span: &str, out: &mut Tokens) {
    if span.is_empty() {
        return;
    }
    out.0.extend(tokenize_text_bpe(vocab, span).0);
}

fn special_token_at(vocab: &Vocabulary, text: &str) -> Option<(i32, usize)> {
    let specials = [
        ("<｜begin▁of▁sentence｜>", vocab.bos_id),
        ("<｜end▁of▁sentence｜>", vocab.eos_id),
        ("<｜User｜>", vocab.user_id),
        ("<｜Assistant｜>", vocab.assistant_id),
        ("<think>", vocab.think_start_id),
        ("</think>", vocab.think_end_id),
        ("｜DSML｜", vocab.dsml_id),
    ];
    for (special, token) in specials {
        if text.starts_with(special) {
            return Some((token, special.len()));
        }
    }
    None
}

fn bpe_emit_piece(vocab: &Vocabulary, raw_piece: &str, out: &mut Tokens) {
    let encoded = byte_encode(raw_piece.as_bytes());
    let mut sym: Vec<String> = encoded.chars().map(|ch| ch.to_string()).collect();

    loop {
        let mut best_i: Option<usize> = None;
        let mut best_rank = i32::MAX;
        for i in 0..sym.len().saturating_sub(1) {
            let s1 = &sym[i];
            let s2 = &sym[i + 1];
            let len = s1.len() + 1 + s2.len();
            
            let rank = if len <= 128 {
                let mut buf: smallvec::SmallVec<[u8; 128]> = smallvec::SmallVec::new();
                buf.extend_from_slice(s1.as_bytes());
                buf.push(b' ');
                buf.extend_from_slice(s2.as_bytes());
                
                if let Ok(key) = std::str::from_utf8(&buf) {
                    vocab.merges.get(key)
                } else {
                    None
                }
            } else {
                let key = format!("{} {}", s1, s2);
                vocab.merges.get(&key)
            };

            if let Some(rank) = rank {
                if *rank < best_rank {
                    best_rank = *rank;
                    best_i = Some(i);
                }
            }
        }

        let Some(best_i) = best_i else { break };
        let merged = format!("{}{}", sym[best_i], sym[best_i + 1]);
        sym[best_i] = merged;
        sym.remove(best_i + 1);
    }

    for symbol in sym {
        if let Some(token) = vocab.token_to_id.get(&symbol) {
            out.push(*token);
            continue;
        }
        for ch in symbol.chars() {
            let one = ch.to_string();
            if let Some(token) = vocab.token_to_id.get(&one) {
                out.push(*token);
            }
        }
    }
}

fn byte_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        if let Some(ch) = char::from_u32(gpt2_byte_to_codepoint(b)) {
            out.push(ch);
        }
    }
    out
}

fn decode_tokens_vocab(vocab: &Vocabulary, tokens: &Tokens) -> String {
    let mut out = Vec::new();
    for token in &tokens.0 {
        let Some(piece) = vocab.tokens.get(*token as usize) else {
            continue;
        };
        if is_control_token(vocab, *token) {
            continue;
        }
        append_decoded_piece(piece, &mut out);
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn append_decoded_piece(piece: &str, out: &mut Vec<u8>) {
    for ch in piece.chars() {
        if let Some(byte) = gpt2_codepoint_to_byte(ch as u32) {
            out.push(byte);
        } else {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
}

fn is_control_token(vocab: &Vocabulary, token: i32) -> bool {
    token == vocab.bos_id
        || token == vocab.eos_id
        || token == vocab.user_id
        || token == vocab.assistant_id
        || token == vocab.think_start_id
        || token == vocab.think_end_id
        || token == vocab.dsml_id
}

fn gpt2_byte_to_codepoint(b: u8) -> u32 {
    if (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174 {
        return b as u32;
    }
    let mut n = 0u32;
    for x in 0u32..=255 {
        if (33..=126).contains(&x) || (161..=172).contains(&x) || x >= 174 {
            continue;
        }
        if x == b as u32 {
            return 256 + n;
        }
        n += 1;
    }
    b as u32
}

fn gpt2_codepoint_to_byte(cp: u32) -> Option<u8> {
    (0u16..=255).find_map(|byte| {
        if gpt2_byte_to_codepoint(byte as u8) == cp {
            Some(byte as u8)
        } else {
            None
        }
    })
}

fn joyai_pieces(text: &str) -> Vec<(usize, usize)> {
    let len = text.len();
    let mut pos = 0usize;
    let mut out = Vec::new();

    while pos < len {
        let start = pos;
        let c = text.as_bytes()[pos];

        if ascii_digit(c) {
            let mut ndigits = 0usize;
            while pos < len && ascii_digit(text.as_bytes()[pos]) && ndigits < 3 {
                pos += 1;
                ndigits += 1;
            }
        } else if joyai_cjk_at(text, pos) {
            loop {
                pos = next_utf8_char(text, pos);
                if pos >= len || !joyai_cjk_at(text, pos) {
                    break;
                }
            }
        } else if joyai_ascii_punct_symbol(c)
            && pos + 1 < len
            && ascii_alpha(text.as_bytes()[pos + 1])
        {
            pos += 1;
            while pos < len && ascii_alpha(text.as_bytes()[pos]) {
                pos += 1;
            }
        } else if joyai_letter_like_at(text, pos) {
            pos = joyai_consume_letters(text, pos);
        } else if !ascii_newline(c)
            && !joyai_ascii_punct_symbol(c)
            && pos + 1 < len
            && joyai_letter_like_at(text, pos + 1)
        {
            pos += 1;
            pos = joyai_consume_letters(text, pos);
        } else if c == b' '
            && pos + 1 < len
            && joyai_ascii_punct_symbol(text.as_bytes()[pos + 1])
        {
            pos += 1;
            while pos < len && joyai_ascii_punct_symbol(text.as_bytes()[pos]) {
                pos += 1;
            }
            while pos < len && ascii_newline(text.as_bytes()[pos]) {
                pos += 1;
            }
        } else if joyai_ascii_punct_symbol(c) {
            while pos < len && joyai_ascii_punct_symbol(text.as_bytes()[pos]) {
                pos += 1;
            }
            while pos < len && ascii_newline(text.as_bytes()[pos]) {
                pos += 1;
            }
        } else if ascii_space(c) {
            let mut p = pos;
            let mut last_newline_end = 0usize;
            while p < len && ascii_space(text.as_bytes()[p]) {
                let sc = text.as_bytes()[p];
                p += 1;
                if ascii_newline(sc) {
                    last_newline_end = p;
                }
            }
            if last_newline_end != 0 {
                pos = last_newline_end;
            } else if p < len
                && p > pos + 1
                && (joyai_letter_like_at(text, p)
                    || joyai_ascii_punct_symbol(text.as_bytes()[p]))
            {
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, pos);
        }

        if pos == start {
            pos = next_utf8_char(text, pos);
        }
        out.push((start, pos));
    }
    out
}

fn next_utf8_char(text: &str, pos: usize) -> usize {
    if pos >= text.len() {
        return pos;
    }
    match text[pos..].chars().next() {
        Some(ch) => pos + ch.len_utf8(),
        None => pos + 1,
    }
}

fn joyai_consume_letters(text: &str, mut pos: usize) -> usize {
    while pos < text.len() && joyai_letter_like_at(text, pos) {
        pos = next_utf8_char(text, pos);
    }
    pos
}

fn joyai_letter_like_at(text: &str, pos: usize) -> bool {
    let b = text.as_bytes()[pos];
    if b < 128 {
        ascii_alpha(b)
    } else {
        true
    }
}

fn joyai_cjk_at(text: &str, pos: usize) -> bool {
    let Some(ch) = text[pos..].chars().next() else {
        return false;
    };
    utf8_is_cjk_hira_kata(ch)
}

fn utf8_is_cjk_hira_kata(ch: char) -> bool {
    matches!(ch as u32, 0x4e00..=0x9fa5 | 0x3040..=0x309f | 0x30a0..=0x30ff)
}

fn ascii_alpha(c: u8) -> bool {
    c.is_ascii_alphabetic()
}

fn ascii_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

fn ascii_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn ascii_newline(c: u8) -> bool {
    matches!(c, b'\n' | b'\r')
}

fn joyai_ascii_punct_symbol(c: u8) -> bool {
    (b'!'..=b'/').contains(&c)
        || (b':'..=b'@').contains(&c)
        || (b'['..=b'`').contains(&c)
        || (b'{'..=b'~').contains(&c)
}

fn required_token_id(token_to_id: &HashMap<String, i32>, text: &str) -> Result<i32> {
    token_to_id
        .get(text)
        .copied()
        .ok_or_else(|| Ds4Error::Protocol(format!("required tokenizer token is missing: {text}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vocab() -> Vocabulary {
        let tokens = vec![
            "<｜begin▁of▁sentence｜>".to_string(),
            "<｜end▁of▁sentence｜>".to_string(),
            "<｜User｜>".to_string(),
            "<｜Assistant｜>".to_string(),
            "<think>".to_string(),
            "</think>".to_string(),
            "｜DSML｜".to_string(),
            "h".to_string(),
            "e".to_string(),
            "l".to_string(),
            "o".to_string(),
            "he".to_string(),
            "ll".to_string(),
            "hello".to_string(),
            "Ġ".to_string(),
            "w".to_string(),
            "r".to_string(),
            "d".to_string(),
            "wo".to_string(),
            "or".to_string(),
            "ld".to_string(),
            "world".to_string(),
        ];
        let mut token_to_id = HashMap::new();
        for (idx, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), idx as i32);
        }
        let mut merges = HashMap::new();
        merges.insert("h e".to_string(), 0);
        merges.insert("l l".to_string(), 1);
        merges.insert("he ll".to_string(), 2);
        merges.insert("hell o".to_string(), 3);
        merges.insert("w o".to_string(), 4);
        merges.insert("wo r".to_string(), 5);
        merges.insert("wor l".to_string(), 6);
        merges.insert("worl d".to_string(), 7);
        Vocabulary {
            tokens,
            token_to_id,
            merges,
            bos_id: 0,
            eos_id: 1,
            user_id: 2,
            assistant_id: 3,
            think_start_id: 4,
            think_end_id: 5,
            dsml_id: 6,
        }
    }

    #[test]
    fn bpe_merges_simple_word() {
        let vocab = test_vocab();
        let tokens = tokenize_text_bpe(&vocab, "hello");
        assert_eq!(tokens.0, vec![13]);
    }

    #[test]
    fn rendered_chat_uses_special_tokens() {
        let vocab = test_vocab();
        let tokens = tokenize_rendered_chat_vocab(&vocab, "<｜User｜>hello<｜Assistant｜><think>");
        assert_eq!(tokens.0, vec![2, 13, 3, 4]);
    }

    #[test]
    fn decodes_byte_level_tokens_back_to_text() {
        let vocab = test_vocab();
        let tokens = Tokens(vec![13, 14, 21]);
        assert_eq!(decode_tokens_vocab(&vocab, &tokens), "hello world");
    }
}
