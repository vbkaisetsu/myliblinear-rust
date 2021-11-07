use std::collections::HashMap;
use std::hash::Hash;

pub struct Model<K> {
    pub(crate) feature_map: HashMap<K, usize>,
    pub(crate) ws: Vec<f64>,
}

impl<K> Model<K>
where
    K: Eq + Hash,
{
    pub fn get_weight(&self, key: &K) -> Option<f64> {
        let id = *self.feature_map.get(key)?;
        self.ws.get(id).cloned()
    }
}
