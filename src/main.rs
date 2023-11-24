use std::collections::HashMap;
use std::{cmp, fs};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use std::time::{Instant};
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
struct Status {
    /// 最大HP
    max_hp: i32,
    /// 攻撃力
    atk: i32,
    /// 防御力
    def: i32,
    /// 運(0~100)
    luc: i32,
    /// 速度
    speed: i32,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
struct Config {
    /// 遺伝子数
    gene_num: i32,
    /// 選択率(0~1.0)
    selection_rate: f32,
    /// 突然変異率(0~1.0)
    mutation_rate: f32,
    /// 最大世代数
    generation_max: i32,
    /// 許容誤差(0~1.0)
    allowable_error: f32,
    /// 最小ターン数
    turn_min: i32,
    /// 最大ターン数
    turn_max: i32,
    /// 防御率
    protection_ratio: f32,
    /// クリティカル時のダメージ倍率
    critical_ratio: f32,
    /// 最小ステータス値
    min_status: Status,
    /// 最大ステータス値
    max_status: Status,
    /// キャラクターのステータス
    character_status: Status,
    /// GAの経過を表示するか
    show_progress: bool,
}

impl Config {
    /// config.jsonからConfigを読み込みます
    fn read_from_json() -> Self {
        let content = fs::read_to_string("config.json")
            .expect("Failed to load config.json");
        serde_json::from_str(&content).unwrap()
    }
}

type Fitness = f32;

const WORST_SCORE: Fitness = 1_000_000.0;

#[derive(Clone, Copy, Debug)]
struct Unit {
    id: Uuid,
    is_monster: bool,
    max_hp: i32,
    hp: i32,
    atk: i32,
    def: i32,
    luc: i32,
    speed: i32,
}

impl Unit {
    fn character_from(status: &Status) -> Unit {
        Unit {
            id: Uuid::new_v4(),
            is_monster: false,
            max_hp: status.max_hp,
            hp: status.max_hp,
            atk: status.atk,
            def: status.def,
            luc: status.luc,
            speed: status.speed,
        }
    }

    fn monster_from(status: &Status) -> Unit {
        Unit {
            id: Uuid::new_v4(),
            is_monster: true,
            max_hp: status.max_hp,
            hp: status.max_hp,
            atk: status.atk,
            def: status.def,
            luc: status.luc,
            speed: status.speed,
        }
    }

    fn to_s(&self) -> String {
        format!("{{ hp={}/{} atk={} def={} luc={} speed={} }}",
                self.hp,
                self.max_hp,
                self.atk,
                self.def,
                self.luc,
                self.speed)
    }
}

#[derive(PartialEq, Debug)]
enum BattleResult {
    Win,
    GameOver,
    Draw,
}

/// # Returns
/// BattleResultと終了ターン数のタプルを返します
fn battle(character: &mut Unit, monster: Unit, config: &Config) -> (BattleResult, i32) {
    let mut rng = rand::thread_rng();

    let mut units: Vec<Unit> = vec![
        character.clone(),
        monster.clone(),
    ];
    units.sort_by(|a, b| b.speed.cmp(&a.speed));    // speed desc

    let mut hp_map: HashMap<Uuid, i32> = HashMap::new();
    for u in units.iter() {
        hp_map.insert(u.id, u.hp);
    }

    let mut turn = 0;
    let result = loop {
        turn += 1;
        if turn > config.turn_max {
            break BattleResult::Draw;
        }

        let mut stop: Option<BattleResult> = None;

        for attacker in units.iter() {
            if *hp_map.get(&attacker.id).unwrap() <= 0 {
                // Already dead
                continue;
            }

            // 攻撃対象
            let defender = if attacker.is_monster {
                character.clone()
            } else {
                monster.clone()
            };

            let mut hp = *hp_map.get(&defender.id).unwrap();
            hp -= calc_damage(&attacker, &defender, &config, &mut rng);
            hp = cmp::max(0, hp);
            hp_map.insert(defender.id, hp);

            if is_dead(&defender.id, &hp_map) {
                if defender.is_monster {
                    stop = Some(BattleResult::Win);
                    break;
                } else {
                    stop = Some(BattleResult::GameOver);
                    break;
                }
            }
        }

        match stop {
            Some(res) => break res,
            None => {}
        };
    };

    let hp = *hp_map.get(&character.id).unwrap();
    character.hp = hp;

    (result, turn)
}

fn is_dead(unit_id: &Uuid, hp_map: &HashMap<Uuid, i32>) -> bool {
    let hp = *hp_map.get(unit_id).unwrap();
    hp <= 0
}

fn calc_damage(attacker: &Unit, defender: &Unit, config: &Config, rng: &mut ThreadRng) -> i32 {
    let luc = attacker.luc as f32 / 100.0;
    let c = if rng.gen::<f32>() <= luc { config.critical_ratio } else { 1.0 };
    let atk = attacker.atk as f32;
    let def = defender.def as f32;
    let damage = ((atk - def * config.protection_ratio) * c).floor() as i32;
    cmp::max(0, damage)
}

struct FitnessFunc {
    config: Config,
}

impl FitnessFunc {
    fn new(config: Config) -> FitnessFunc {
        FitnessFunc {
            config,
        }
    }

    pub fn calc(&self, monster_status: &Status) -> (Fitness, BattleResult) {
        let mut character = Unit::character_from(&self.config.character_status);
        let monster = Unit::monster_from(&monster_status);

        let (result, turn) = battle(&mut character, monster, &self.config);

        // 残りHPの割合
        let hp_ratio = character.hp as f32 / character.max_hp as f32;

        // 適合度
        let score = if result == BattleResult::GameOver {
            WORST_SCORE
        } else if result == BattleResult::Draw {
            WORST_SCORE - 1.0
        } else if turn <= self.config.turn_min {
            WORST_SCORE - 1.0
        } else {
            hp_ratio
        };

        if self.config.show_progress {
            println!("-------------------------------");
            println!("score={} {:?} turn={}", score, result, turn);
            println!("character={}", character.to_s());
            println!("monster={}", monster.to_s());
            println!("-------------------------------");
        }

        (score, result)
    }
}

/// GAで求めるパラメータセット
type GeneParams = Status;
/// パラメータが取りうる範囲の最大値を格納する構造体
type MaxValuesOfStatus = GeneParams;
/// パラメータが取りうる範囲の最小値を格納する構造体
type MinValuesOfStatus = GeneParams;

#[derive(Clone, Copy, Debug)]
struct Gene {
    id: Uuid,
    fitness: Fitness,
    params: GeneParams,
}

impl Gene {
    fn new(params: GeneParams) -> Self {
        Gene {
            id: Uuid::new_v4(),
            fitness: WORST_SCORE,
            params,
        }
    }

    fn to_s(&self) -> String {
        format!("fitness={} {{ max_hp={} atk={} def={} luc={} speed={} }}",
                self.fitness,
                self.params.max_hp,
                self.params.atk,
                self.params.def,
                self.params.luc,
                self.params.speed)
    }
}

struct GeneMask {
    max_hp: bool,
    atk: bool,
    def: bool,
    luc: bool,
    speed: bool,
}

impl GeneMask {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        GeneMask {
            max_hp: rng.gen(),
            atk: rng.gen(),
            def: rng.gen(),
            luc: rng.gen(),
            speed: rng.gen(),
        }
    }
}

struct Mutation {
    min: MinValuesOfStatus,
    max: MaxValuesOfStatus,
}

impl Mutation {
    /// ランダムに遺伝子を生成します
    fn new_at_random(&self) -> Gene {
        let mut rng = rand::thread_rng();
        let hp = Uniform::from(self.min.max_hp..=self.max.max_hp);
        let atk = Uniform::from(self.min.atk..=self.max.atk);
        let def = Uniform::from(self.min.def..=self.max.def);
        let luc = Uniform::from(self.min.luc..=self.max.luc);
        let speed = Uniform::from(self.min.speed..=self.max.speed);

        let params = GeneParams {
            max_hp: hp.sample(&mut rng),
            atk: atk.sample(&mut rng),
            def: def.sample(&mut rng),
            luc: luc.sample(&mut rng),
            speed: speed.sample(&mut rng),
        };
        Gene::new(params)
    }

    /// 突然変異させます
    fn mutate(&self, gene: &mut Gene) {
        let mut rng = rand::thread_rng();
        let hp = Uniform::from(self.min.max_hp..=self.max.max_hp);
        let atk = Uniform::from(self.min.atk..=self.max.atk);
        let def = Uniform::from(self.min.def..=self.max.def);
        let luc = Uniform::from(self.min.luc..=self.max.luc);
        let speed = Uniform::from(self.min.speed..=self.max.speed);
        let mask: GeneMask = GeneMask::new();

        if mask.max_hp {
            gene.params.max_hp = hp.sample(&mut rng);
        }
        if mask.atk {
            gene.params.atk = atk.sample(&mut rng);
        }
        if mask.def {
            gene.params.def = def.sample(&mut rng);
        }
        if mask.luc {
            gene.params.luc = luc.sample(&mut rng);
        }
        if mask.speed {
            gene.params.speed = speed.sample(&mut rng);
        }
    }
}

struct GeneticAlgorithm {
    config: Config,
    mutation: Mutation,
    genes: Vec<Gene>,
}

impl GeneticAlgorithm {
    ///
    /// # Arguments
    /// * `config` - 設定
    /// * `mutation` - 突然変異の実装
    fn new(config: Config, mutation: Mutation) -> Self {
        let genes: Vec<Gene> = vec![];
        GeneticAlgorithm {
            config,
            mutation,
            genes,
        }
    }

    /// 次世代に残す遺伝子の数を返します
    fn selection_num(&self) -> usize {
        (self.config.gene_num as f32 * self.config.selection_rate).floor() as usize
    }

    /// 許容誤差を返します
    fn allowable_score(&self) -> Fitness {
        self.config.allowable_error
    }

    /// ベストスコアを返します
    fn best_score(&self) -> Fitness {
        self.genes[0].fitness
    }

    /// 先頭から指定個数分の遺伝子を返します
    fn head(&self, num: usize) -> Vec<Gene> {
        (&self.genes[..num]).to_vec()
    }

    /// ランダムで選ばれた1つの遺伝子を返します
    fn sample(&self) -> Gene {
        let mut rng = rand::thread_rng();
        let idx = Uniform::from(0..self.genes.len());
        self.genes[idx.sample(&mut rng)].clone()
    }

    /// アルゴリズムを実行します
    ///
    /// # Returns
    /// 世代数を返します
    fn exec(&mut self) -> i32 {
        let gene_num = self.config.gene_num as usize;

        // 次世代に残す遺伝子の数
        let selection_num = self.selection_num();

        // 適合度がこの誤差内に収まるときループを早期終了
        let allowable_score = self.allowable_score();

        // 初期世代の生成
        for _ in 0..gene_num {
            self.genes.push(self.mutation.new_at_random());
        }

        let ff = FitnessFunc::new(self.config.clone());

        // 適合度計算
        for gene in self.genes.iter_mut() {
            gene.fitness = ff.calc(&gene.params).0;
        }

        // 現在のベストスコア
        let mut cur_best_score = WORST_SCORE;
        // 現在の世代
        let mut generation = 1;
        loop {
            // 適合度降順(優良順)にソート
            self.genes.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

            let best_score = self.best_score();
            if cur_best_score != best_score {
                cur_best_score = best_score;
                println!("{}G: {}", generation, best_score);
            }

            if generation >= self.config.generation_max {
                // Finish
                break;
            } else if best_score <= allowable_score {
                println!("early stopping!");
                break;
            }

            // 選択
            // 優良個体を次世代に残す
            let mut new_genes = self.head(selection_num);

            // 選択されなかった個数分、交叉または突然変異により生成する
            let mut rng = rand::thread_rng();
            loop {
                // rng.gen::<f32>() -> 0 <= p < 1.0
                if rng.gen::<f32>() <= self.config.mutation_rate {
                    // 突然変異
                    let mut g = self.sample();
                    self.mutation.mutate(&mut g);

                    // 適合度計算
                    g.fitness = ff.calc(&g.params).0;
                    new_genes.push(g);
                    if new_genes.len() == gene_num {
                        break;
                    }
                } else {
                    // 両親の選択
                    let parents = self.select_parents();
                    // 交叉
                    let children = self.crossover(&parents.0, &parents.1);

                    // 子1
                    let mut g1 = children.0;
                    // 適合度計算
                    g1.fitness = ff.calc(&g1.params).0;
                    new_genes.push(g1);
                    if new_genes.len() == gene_num {
                        break;
                    }

                    // 子2
                    let mut g2 = children.1;
                    // 適合度計算
                    g2.fitness = ff.calc(&g2.params).0;
                    new_genes.push(g2);
                    if new_genes.len() == gene_num {
                        break;
                    }
                }
            }
            self.genes = new_genes;
            assert_eq!(self.genes.len(), gene_num);

            generation += 1;
        }

        generation
    }

    /// ランダムに親を選出します
    fn select_parents(&self) -> (Gene, Gene) {
        let p1: Gene = self.sample();
        let p2 = loop {
            let p2: Gene = self.sample();
            if p1.id != p2.id {
                break p2;
            }
        };
        (p1, p2)
    }

    /// 親1と親2を交叉して新しい遺伝子を生成します
    ///
    /// # Arguments
    /// * `p1` - Parent 1
    /// * `p2` - Parent 2
    fn crossover(&self, p1: &Gene, p2: &Gene) -> (Gene, Gene) {
        let mask: GeneMask = GeneMask::new();

        // 一様交叉
        let params1 = GeneParams {
            max_hp: if mask.max_hp { p2.params.max_hp } else { p1.params.max_hp },
            atk: if mask.atk { p2.params.atk } else { p1.params.atk },
            def: if mask.def { p2.params.def } else { p1.params.def },
            luc: if mask.luc { p2.params.luc } else { p1.params.luc },
            speed: if mask.speed { p2.params.speed } else { p1.params.speed },
        };
        let params2 = GeneParams {
            max_hp: if mask.max_hp { p1.params.max_hp } else { p2.params.max_hp },
            atk: if mask.atk { p1.params.atk } else { p2.params.atk },
            def: if mask.def { p1.params.def } else { p2.params.def },
            luc: if mask.luc { p1.params.luc } else { p2.params.luc },
            speed: if mask.speed { p1.params.speed } else { p2.params.speed },
        };
        (Gene::new(params1), Gene::new(params2))
    }
}

fn main() {
    let config = Config::read_from_json();

    let mut ga = GeneticAlgorithm::new(config.clone(),
                                       Mutation {
                                           min: config.min_status.clone(),
                                           max: config.max_status.clone(),
                                       });

    let start = Instant::now();

    let generation = ga.exec();

    let end = start.elapsed();

    // 最高スコアのパラメータでテストプレイ
    let best_gene: Gene = ga.head(1)[0];
    let ff = FitnessFunc::new(config);
    let mut win_count = 0;
    let simulation_count = 100;
    for _ in 0..simulation_count {
        let (_, result) = ff.calc(&best_gene.params);
        if result == BattleResult::Win {
            win_count += 1;
        }
    }

    println!("elapsed={}.{:03}sec", end.as_secs(), end.subsec_nanos() / 1_000_000);
    println!("generation={}\n{:#?}",
             generation,
             ga.head(5).iter().map(|g| g.to_s()).collect::<Vec<String>>()
    );
    println!("win={}%", win_count as f32 / simulation_count as f32 * 100.0);
}
