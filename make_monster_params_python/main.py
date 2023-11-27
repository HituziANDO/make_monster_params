from enum import Enum
from typing import Tuple
import json
import random
import time
import uuid


class Status:
    """ Statusクラスの定義 """

    def __init__(self, max_hp: int, atk: int, def_: int, luc: int, speed: int):
        """ コンストラクタ """
        self.max_hp = max_hp  # 最大HP
        self.atk = atk  # 攻撃力
        self.def_ = def_  # 防御力
        self.luc = luc  # 運(0~100)
        self.speed = speed  # 速度


class Config:
    """ Configクラスの定義 """

    def __init__(self,
                 gene_num: int,
                 selection_rate: float,
                 mutation_rate: float,
                 generation_max: int,
                 turn_min: int,
                 turn_max: int,
                 protection_ratio: float,
                 critical_ratio: float,
                 min_status: dict,
                 max_status: dict,
                 character_status: dict,
                 show_progress: bool):
        """ コンストラクタ """
        self.gene_num = gene_num  # 遺伝子数
        self.selection_rate = selection_rate  # 選択率(0~1.0)
        self.mutation_rate = mutation_rate  # 突然変異率(0~1.0)
        self.generation_max = generation_max  # 最大世代数
        self.turn_min = turn_min  # 最小ターン数
        self.turn_max = turn_max  # 最大ターン数
        self.protection_ratio = protection_ratio  # 防御率
        self.critical_ratio = critical_ratio  # クリティカル時のダメージ倍率
        self.min_status = self.__convert_dict_to_status(min_status)  # 最小ステータス値
        self.max_status = self.__convert_dict_to_status(max_status)  # 最大ステータス値
        self.character_status = self.__convert_dict_to_status(character_status)  # キャラクターのステータス
        self.show_progress = show_progress  # GAの経過を表示するか

    @classmethod
    def read_from_json(cls):
        """ config.jsonからConfigを読み込みます """
        with open('config.json', 'r') as f:
            data = json.load(f)
        return cls(**data)

    @staticmethod
    def __convert_dict_to_status(dict_: dict) -> Status:
        converted_status = {}
        for key, value in dict_.items():
            if key.startswith("def") and value is not None:
                converted_status[key.replace("def", "def_")] = value
            else:
                converted_status[key] = value
        return Status(**converted_status)


Fitness = float

WORST_SCORE = 1_000_000.0


class Unit:
    def __init__(self, is_monster: bool, status: Status):
        self.id = uuid.uuid4()
        self.is_monster = is_monster
        self.max_hp = status.max_hp
        self.hp = status.max_hp
        self.atk = status.atk
        self.def_ = status.def_
        self.luc = status.luc
        self.speed = status.speed

    @classmethod
    def character_from(cls, status: Status):
        return cls(False, status)

    @classmethod
    def monster_from(cls, status: Status):
        return cls(True, status)

    def to_s(self) -> str:
        return f"{{ hp={self.hp}/{self.max_hp} atk={self.atk} def={self.def_} luc={self.luc} speed={self.speed} }}"


class BattleResult(Enum):
    WIN = 1
    GAME_OVER = 2
    DRAW = 3


def battle(character: Unit, monster: Unit, config: Config) -> Tuple[BattleResult, int]:
    units = sorted([character, monster], key=lambda u: u.speed, reverse=True)

    hp_map = {u.id: u.hp for u in units}

    turn = 0
    while True:
        turn += 1
        if turn > config.turn_max:
            result = BattleResult.DRAW
            break

        stop = None
        for attacker in units:
            if hp_map[attacker.id] <= 0:
                continue

            defender = character if attacker.is_monster else monster

            hp = hp_map[defender.id]
            hp -= calc_damage(attacker, defender, config)
            hp = max(0, hp)
            hp_map[defender.id] = hp

            if hp_map[defender.id] <= 0:
                if defender.is_monster:
                    stop = BattleResult.WIN
                    break
                else:
                    stop = BattleResult.GAME_OVER
                    break

        if stop is not None:
            result = stop
            break

    character.hp = hp_map[character.id]

    return result, turn


def calc_damage(attacker: Unit, defender: Unit, config: Config) -> int:
    luc = attacker.luc / 100.0
    c = config.critical_ratio if random.random() <= luc else 1.0
    atk = attacker.atk
    def_ = defender.def_
    damage = int((atk - def_ * config.protection_ratio) * c)
    return max(0, damage)


class FitnessFunc:
    def __init__(self, config: Config):
        self.config = config

    def calc(self, monster_status: Status) -> Tuple[Fitness, BattleResult]:
        character = Unit.character_from(self.config.character_status)
        monster = Unit.monster_from(monster_status)

        result, turn = battle(character, monster, self.config)

        # 残りHPの割合
        hp_ratio = float(character.hp) / float(character.max_hp)

        # 適合度
        if result == BattleResult.GAME_OVER:
            score = WORST_SCORE
        elif result == BattleResult.DRAW:
            score = WORST_SCORE - 1.0
        elif turn < self.config.turn_min:
            score = WORST_SCORE - 1.0
        else:
            score = hp_ratio

        if self.config.show_progress:
            print("-------------------------------")
            print(f"score={score} {result} turn={turn}")
            print(f"character={character.to_s()}")
            print(f"monster={monster.to_s()}")
            print("-------------------------------")

        return score, result


# GAで求めるパラメータセット
GeneParams = Status


class Gene:
    def __init__(self, params: GeneParams):
        self.id = uuid.uuid4()
        self.fitness = WORST_SCORE
        self.params = params

    def to_s(self) -> str:
        return (f"fitness={self.fitness} "
                f"{{ max_hp={self.params.max_hp} atk={self.params.atk} def={self.params.def_} luc={self.params.luc} speed={self.params.speed} }}")


class GeneMask:
    def __init__(self):
        self.max_hp = random.choice([True, False])
        self.atk = random.choice([True, False])
        self.def_ = random.choice([True, False])
        self.luc = random.choice([True, False])
        self.speed = random.choice([True, False])


class Mutation:
    def __init__(self, min: Status, max: Status):
        self.min = min
        self.max = max

    def new_at_random(self) -> Gene:
        params = GeneParams(
            max_hp=random.randint(self.min.max_hp, self.max.max_hp),
            atk=random.randint(self.min.atk, self.max.atk),
            def_=random.randint(self.min.def_, self.max.def_),
            luc=random.randint(self.min.luc, self.max.luc),
            speed=random.randint(self.min.speed, self.max.speed)
        )
        return Gene(params)

    def mutated(self, gene: Gene, rate: float) -> Gene:
        return Gene(GeneParams(
            max_hp=gene.params.max_hp if random.random() > rate else random.randint(self.min.max_hp, self.max.max_hp),
            atk=gene.params.atk if random.random() > rate else random.randint(self.min.atk, self.max.atk),
            def_=gene.params.def_ if random.random() > rate else random.randint(self.min.def_, self.max.def_),
            luc=gene.params.luc if random.random() > rate else random.randint(self.min.luc, self.max.luc),
            speed=gene.params.speed if random.random() > rate else random.randint(self.min.speed, self.max.speed)
        ))


class GeneticAlgorithm:
    def __init__(self, config: Config, mutation: Mutation):
        self.config = config
        self.mutation = mutation
        self.genes = []

    def selection_num(self) -> int:
        return int(self.config.gene_num * self.config.selection_rate)

    def best_score(self) -> Fitness:
        return self.genes[0].fitness

    def head(self, num) -> list:
        return self.genes[:num]

    def sample(self) -> Gene:
        return random.choice(self.genes)

    def exec(self) -> int:
        gene_num = self.config.gene_num

        # 次世代に残す遺伝子の数
        selection_num = self.selection_num()

        # 初期世代の生成
        for _ in range(gene_num):
            self.genes.append(self.mutation.new_at_random())

        ff = FitnessFunc(self.config)

        # 適合度計算
        for gene in self.genes:
            gene.fitness = ff.calc(gene.params)[0]

        # 現在のベストスコア
        cur_best_score = WORST_SCORE
        # 現在の世代
        generation = 1
        while True:
            # 適合度降順(優良順)にソート (優良順はfitnessの値が小さい順)
            self.genes = sorted(self.genes, key=lambda gene: gene.fitness)

            best_score = self.best_score()
            if cur_best_score != best_score:
                cur_best_score = best_score
                print(f"{generation}G: {best_score}")

            if generation >= self.config.generation_max:
                # Finish
                break

            # 選択
            # 優良個体を次世代に残す
            new_genes = self.head(selection_num)

            # 選択されなかった個数分、交叉・突然変異により生成する
            while len(new_genes) < gene_num:
                # 両親の選択
                parents = self.select_parents()
                # 交叉
                children = self.crossover(parents[0], parents[1])

                # 子1
                # 突然変異
                g1 = self.mutation.mutated(children[0], self.config.mutation_rate)
                # 適合度計算
                g1.fitness = ff.calc(g1.params)[0]
                new_genes.append(g1)
                if len(new_genes) == gene_num:
                    break

                # 子2
                # 突然変異
                g2 = self.mutation.mutated(children[1], self.config.mutation_rate)
                # 適合度計算
                g2.fitness = ff.calc(g2.params)[0]
                new_genes.append(g2)
                if len(new_genes) == gene_num:
                    break

            self.genes = new_genes

            generation += 1

        return generation

    def select_parents(self):
        p1 = self.sample()
        p2 = p1
        while p1.id == p2.id:
            p2 = self.sample()
        return p1, p2

    def crossover(self, p1, p2):
        mask = GeneMask()

        params1 = GeneParams(
            max_hp=p2.params.max_hp if mask.max_hp else p1.params.max_hp,
            atk=p2.params.atk if mask.atk else p1.params.atk,
            def_=p2.params.def_ if mask.def_ else p1.params.def_,
            luc=p2.params.luc if mask.luc else p1.params.luc,
            speed=p2.params.speed if mask.speed else p1.params.speed
        )
        params2 = GeneParams(
            max_hp=p1.params.max_hp if mask.max_hp else p2.params.max_hp,
            atk=p1.params.atk if mask.atk else p2.params.atk,
            def_=p1.params.def_ if mask.def_ else p2.params.def_,
            luc=p1.params.luc if mask.luc else p2.params.luc,
            speed=p1.params.speed if mask.speed else p2.params.speed
        )
        return Gene(params1), Gene(params2)


def main():
    config = Config.read_from_json()

    ga = GeneticAlgorithm(config, Mutation(min=config.min_status, max=config.max_status))

    start = time.time()

    generation = ga.exec()

    end = time.time() - start

    # 最高スコアのパラメータでテストプレイ
    best_gene = ga.head(1)[0]
    ff = FitnessFunc(config)
    win_count = 0
    simulation_count = 100
    for _ in range(simulation_count):
        _, result = ff.calc(best_gene.params)
        if result == BattleResult.WIN:
            win_count += 1

    print(f"elapsed={end:.3f}sec")
    print(f"generation={generation}\n{[g.to_s() for g in ga.head(5)]}")
    print(f"win={win_count / simulation_count * 100.0}%")


if __name__ == "__main__":
    main()
