# Ensemble classification: Census Income

Классификация Census Income техниками ансамблей: бэггинг, случайный лес и
AdaBoost. В `data/` уже помещены исходные файлы UCI Adult.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Один запуск любой техники
python ensemble.py --data data/adult.data --technique random_forest --n-estimators 100

# Полный эксперимент: 50, 60, 70, 80, 90, 100 участников
python experiments.py --data data/adult.data --technique random_forest
python make_report.py
```

Параметры `ensemble.py`: `--technique {bagging,random_forest,boosting}`,
`--n-estimators`, `--max-depth`, `--max-samples` (для бэггинга),
`--learning-rate` (для бустинга), `--train-fraction` и `--random-state`.

Эксперименты используют то же разбиение 80:20 и seed 42, что и предыдущее
задание с деревом решений; значения его метрик считываются из
`../classication-by-desicion-tree/results/metrics.json` и наносятся на график.
