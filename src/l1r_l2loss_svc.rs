use std::collections::HashMap;
use std::hash::Hash;

use rand::seq::SliceRandom;

use crate::model::Model;
use crate::problem::{Feature, Problem};

enum Label {
    Positive,
    Negative,
}

#[derive(Clone)]
struct TransposedFeature {
    id: usize,
    value: f64,
}

pub struct L1rL2lossSvcSolver {
    eps: f64,
    cost_p: f64,
    cost_n: f64,
    beta: f64,
    sigma: f64,
    max_linesearch_iter: usize,
}

impl L1rL2lossSvcSolver {
    pub fn new(
        eps: f64,
        cost_p: f64,
        cost_n: f64,
        beta: f64,
        sigma: f64,
        max_linesearch_iter: usize,
    ) -> Self {
        Self {
            eps,
            cost_p,
            cost_n,
            beta,
            sigma,
            max_linesearch_iter,
        }
    }
}

pub struct L1rL2lossSvcSolverIter<'a, K> {
    solver: &'a L1rL2lossSvcSolver,
    ws: Option<Vec<f64>>,
    rng: rand::rngs::ThreadRng,
    n_train_features: usize,
    xs_t: Vec<Vec<TransposedFeature>>,
    bs: Vec<f64>,
    ys: Vec<Label>,
    prev_v_max: f64,
    indices: Vec<usize>,
    xs_sq: Vec<f64>,
    v_sum_init: Option<f64>,
    feature_map: Option<HashMap<K, usize>>,
}

impl<'a, K> L1rL2lossSvcSolverIter<'a, K> {
    fn calculate_loss(&self, j: usize) -> f64 {
        let mut loss = 0.0;
        for feat in &self.xs_t[j] {
            if self.bs[feat.id] > 0.0 {
                let cost = match self.ys[feat.id] {
                    Label::Positive => self.solver.cost_p,
                    Label::Negative => self.solver.cost_n,
                };
                loss += cost * self.bs[feat.id].powi(2);
            }
        }
        loss
    }

    fn calculate_g_loss(&self, j: usize) -> (f64, f64) {
        let mut g_loss = 0.0;
        let mut g2_loss = 0.0;
        for feat in &self.xs_t[j] {
            if self.bs[feat.id] > 0.0 {
                let cost = match self.ys[feat.id] {
                    Label::Positive => self.solver.cost_p,
                    Label::Negative => self.solver.cost_n,
                };
                g_loss -= cost * feat.value * self.bs[feat.id];
                g2_loss += cost * feat.value.powi(2);
            }
        }
        (g_loss * 2.0, (g2_loss * 2.0).max(1e-12))
    }
}

impl<'a, K> Iterator for L1rL2lossSvcSolverIter<'a, K> {
    type Item = Option<Model<K>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut v_max = 0.0;
        let mut v_sum = 0.0;
        let mut opt_ws = self.ws.take();
        let ws = opt_ws.as_mut()?;

        self.indices
            .partial_shuffle(&mut self.rng, self.n_train_features);

        let mut s = 0;
        while s < self.n_train_features {
            let j = self.indices[s];

            let (g_loss, g2_loss) = self.calculate_g_loss(j);

            // If optimal
            if ws[j] == 0.0 {
                let m = self.prev_v_max / self.ys.len() as f64;
                if m - 1.0 <= g_loss && g_loss < 1.0 - m {
                    self.n_train_features -= 1;
                    self.indices.swap(s, self.n_train_features);
                    continue;
                }
            }

            s += 1;

            // Calculates violation
            let v = if ws[j] > 0.0 {
                (g_loss + 1.0).abs()
            } else if ws[j] < 0.0 {
                (g_loss - 1.0).abs()
            } else {
                (g_loss - 1.0).max(-1.0 - g_loss).max(0.0)
            };
            v_max = v.max(v_max);
            v_sum += v;

            // Calculates Newton direction
            let mut d = if g_loss + 1.0 < g2_loss * ws[j] {
                -(g_loss + 1.0) / g2_loss
            } else if (g_loss - 1.0) > g2_loss * ws[j] {
                -(g_loss - 1.0) / g2_loss
            } else {
                -ws[j]
            };

            // Conducts a line search procedure
            if d.abs() < 1e-12 {
                continue;
            }
            let prev_loss = self.calculate_loss(j);
            let mut delta = g_loss.mul_add(d, (ws[j] + d).abs() - ws[j].abs());
            let mut prev_d = 0.0;
            let mut num_linesearch = 0;
            for i in 0..self.solver.max_linesearch_iter {
                num_linesearch = i + 1;

                let d_diff = prev_d - d;
                for feat in &self.xs_t[j] {
                    self.bs[feat.id] += d_diff * feat.value;
                }

                // Eq. 42
                if g_loss.mul_add(
                    d,
                    (self.xs_sq[j] * d).mul_add(d, (ws[j] + d).abs() - ws[j].abs()),
                ) <= self.solver.sigma * delta
                {
                    break;
                }

                // Eq. 41
                let loss = self.calculate_loss(j);
                if (ws[j] + d).abs() - ws[j].abs() + loss - prev_loss <= self.solver.sigma * delta {
                    break;
                }

                prev_d = d;
                d *= self.solver.beta;
                delta *= self.solver.beta;
            }

            ws[j] += d;

            // Corrects for errors of b.
            if num_linesearch >= self.solver.max_linesearch_iter {
                self.bs.fill(1.0);
                for (xs_j, &w) in self.xs_t.iter().zip(ws.iter()) {
                    if w == 0.0 {
                        continue;
                    }
                    for feat in xs_j {
                        self.bs[feat.id] -= w * feat.value;
                    }
                }
            }
        }

        if self.v_sum_init.is_none() {
            self.v_sum_init.replace(v_sum);
        }

        if v_sum <= self.v_sum_init.unwrap() * self.solver.eps {
            if self.n_train_features == ws.len() {
                Some(Some(Model {
                    feature_map: self.feature_map.take().unwrap(),
                    ws: opt_ws.unwrap(),
                }))
            } else {
                self.n_train_features = ws.len();
                self.prev_v_max = std::f64::MAX;
                self.ws.replace(opt_ws.unwrap());
                Some(None)
            }
        } else {
            self.prev_v_max = v_max;
            self.ws.replace(opt_ws.unwrap());
            Some(None)
        }
    }
}

impl L1rL2lossSvcSolver {
    fn transpose_xs<K>(
        xs: &[Vec<Feature<K>>],
        ys: &[Label],
        n_features: usize,
    ) -> (HashMap<K, usize>, Vec<Vec<TransposedFeature>>)
    where
        K: Eq + Hash + Clone,
    {
        let mut xs_t = vec![vec![]; n_features];
        let mut feature_map = HashMap::new();
        for (i, (x, y)) in xs.iter().zip(ys).enumerate() {
            let y = match y {
                Label::Positive => 1.0,
                Label::Negative => -1.0,
            };
            for feat in x {
                let feature_id = if let Some(id) = feature_map.get(&feat.key) {
                    *id
                } else {
                    let id = feature_map.len();
                    feature_map.insert(feat.key.clone(), id);
                    id
                };
                xs_t[feature_id].push(TransposedFeature {
                    id: i,
                    value: y * feat.value,
                });
            }
        }
        (feature_map, xs_t)
    }

    pub fn solve<K>(&self, prob: &Problem<K>) -> L1rL2lossSvcSolverIter<K>
    where
        K: Eq + Hash + Clone,
    {
        let n_features = prob.n_features;
        let ws = vec![0.0; n_features];
        let bs = vec![1.0; prob.ys.len()];

        let indices: Vec<_> = (0..n_features).collect();
        let ys: Vec<_> = prob
            .ys
            .iter()
            .map(|y| {
                if *y > 0.0 {
                    Label::Positive
                } else {
                    Label::Negative
                }
            })
            .collect();

        let (feature_map, xs_t) = Self::transpose_xs(&prob.xs, &ys, n_features);

        let mut xs_sq = vec![0.0; n_features];
        for j in 0..n_features {
            xs_sq[j] = 0.0;
            for feat in &xs_t[j] {
                let cost = match ys[feat.id] {
                    Label::Positive => self.cost_p,
                    Label::Negative => self.cost_n,
                };
                xs_sq[j] += cost * feat.value.powi(2);
            }
        }

        L1rL2lossSvcSolverIter {
            solver: self,
            ws: Some(ws),
            rng: rand::thread_rng(),
            n_train_features: prob.n_features,
            xs_t,
            bs,
            ys,
            prev_v_max: std::f64::MAX,
            indices,
            xs_sq,
            v_sum_init: None,
            feature_map: Some(feature_map),
        }
    }
}
