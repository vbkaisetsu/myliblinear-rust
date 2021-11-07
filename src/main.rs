mod l1r_l2loss_svc;
mod model;
mod problem;

use crate::l1r_l2loss_svc::L1rL2lossSvcSolver;
use crate::problem::{Feature, Problem};

use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut xs_train = vec![];
    let mut ys_train = vec![];
    let mut xs_test = vec![];
    let mut ys_test = vec![];

    let mut n_features = 0;

    for (i, line) in io::stdin().lock().lines().enumerate() {
        if i % 1000 == 0 {
            eprintln!("{}", i);
        }
        let line = line?;
        let mut feats = vec![];
        let mut spl = line.split(' ');
        let y: f64 = spl.next().unwrap().parse()?;
        for s in spl {
            if s.is_empty() {
                break;
            }
            let mut spl = s.split(':');
            let id: usize = spl.next().unwrap().parse()?;
            let value: f64 = spl.next().unwrap().parse()?;
            feats.push(Feature::new(id, value));
            n_features = n_features.max(id + 1);
        }
        if i < 4000 {
            xs_train.push(feats);
            ys_train.push(y);
        } else {
            xs_test.push(feats);
            ys_test.push(y);
        }
    }

    let prob = Problem {
        n_features,
        xs: xs_train,
        ys: ys_train,
    };

    let solver = L1rL2lossSvcSolver::new(1e-2, 1.0, 1.0, 0.5, 1e-2, 20);

    let mut model = None;
    for (i, m) in solver.solve(&prob).enumerate() {
        if (i + 1) % 10 == 0 {
            eprint!(".");
            io::stderr().flush().unwrap();
        }
        model = m;
    }
    eprint!("\n");
    let model = model.unwrap();

    let mut tpos = 0;
    let mut fpos = 0;
    let mut fneg = 0;
    for (x, &y) in xs_test.iter().zip(&ys_test) {
        let mut score = 0.0;
        for feat in x {
            if let Some(w) = model.get_weight(&feat.key) {
                score += w * feat.value;
            }
        }
        if y > 0.0 {
            if score > 0.0 {
                tpos += 1;
            } else {
                fneg += 1;
            }
        } else if score > 0.0 {
            fpos += 1;
        }
    }
    let prec = tpos as f64 / (tpos + fpos) as f64;
    let recall = tpos as f64 / (tpos + fneg) as f64;
    println!("prec: {}", prec);
    println!("recall: {}", recall);
    println!("f1: {}", 2.0 * prec * recall / (prec + recall));

    let mut n_nonzero = 0;
    for &r in &model.ws {
        if r != 0.0 {
            n_nonzero += 1;
        }
    }
    dbg!(n_nonzero);

    Ok(())
}
