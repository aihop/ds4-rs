#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tokens(pub Vec<i32>);

impl Tokens {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn push(&mut self, token: i32) {
        self.0.push(token);
    }

    pub fn starts_with(&self, prefix: &Self) -> bool {
        self.0.starts_with(&prefix.0)
    }

    pub fn common_prefix_len(&self, other: &Self) -> usize {
        self.0
            .iter()
            .zip(other.0.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }

    pub fn slice_prefix(&self, len: usize) -> Self {
        Self(self.0.iter().copied().take(len).collect())
    }
}

impl From<Vec<i32>> for Tokens {
    fn from(value: Vec<i32>) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, Default)]
pub struct TokenScore {
    pub id: i32,
    pub logit: f32,
    pub logprob: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionSnapshot {
    pub bytes: Vec<u8>,
}
