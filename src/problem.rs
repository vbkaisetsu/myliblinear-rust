#[derive(Clone)]
pub struct Feature<K> {
    pub(crate) key: K,
    pub(crate) value: f64,
}

impl<K> Feature<K> {
    pub const fn new(key: K, value: f64) -> Self {
        Self { key, value }
    }
}

pub struct Problem<K> {
    pub(crate) n_features: usize,
    pub(crate) xs: Vec<Vec<Feature<K>>>,
    pub(crate) ys: Vec<f64>,
}
