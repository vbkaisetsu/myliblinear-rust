#[derive(Clone, Debug)]
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
    pub(crate) xs: Vec<Vec<Feature<K>>>,
    pub(crate) ys: Vec<f64>,
}

impl<K> Problem<K> {
    pub fn new(xs: Vec<Vec<Feature<K>>>, ys: Vec<f64>) -> Self {
        Self {
            xs,
            ys,
        }
    }
}
