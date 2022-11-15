import pandas as pd
import sqlite3
import numpy as np
import operator
import typing
import dataclasses
import os
import json


@dataclasses.dataclass
class DetailedNextActivityResult:
    id: str
    output: typing.List[typing.Tuple[str, float]]

    def to_dict(self) -> typing.Dict[str, typing.Union[str, typing.List[typing.Tuple[str, float]]]]:
        return {
            'id': self.id,
            'output': self.output
        }

swapped_dict = {0: 'Assign seriousness', 1: 'Take in charge ticket', 2: 'Resolve ticket', 3: 'Closed', 4: 'Insert ticket', 5: 'Wait', 6: 'Create SW anomaly', 7: 'Require upgrade', 8: 'VERIFIED', 9: 'DUPLICATE', 10: 'Resolve SW anomaly', 11: 'Schedule intervention', 12: 'RESOLVED', 13: 'INVALID'}
y_pred = [[-4.461381,   1.1387295,  2.2169352,  5.5290427, -6.8009515, -2.5174925,
  -3.5927484, -7.1931047, -5.737158,  -7.758201,  -5.7513943, -7.7208667,
  -4.845661,  -5.8412085]]

print(y_pred[0])
print(np.argmax(y_pred[0]))
print(swapped_dict[np.argmax(y_pred[0])])

line_shape = [swapped_dict[i] for i in range(0, len(swapped_dict.keys()))]
print(line_shape)

res = zip(y_pred[0], line_shape)
listz = [i for i in zip(line_shape, y_pred[0], )]
for i, j in res:
    print(i, j)

print(listz)
listz.sort(key=lambda x: x[1], reverse=True)

listz.sort(key=operator.itemgetter(1), reverse=True)
print(listz)

print([i[0] for i in listz[:3]])

bla = []
bla.append(DetailedNextActivityResult('123', listz))
with open(os.path.join('D:\\', 'detailed_next_activity_results.jsonl'), 'w', encoding='utf8') as f:
    for res in bla:
        line = json.dumps(res.to_dict())
        f.write(f'{line}\n')



exit(-1)

try:
    conn = sqlite3.connect(r"D:\Appsalot\hybrid-scalar.db")
except Exception as e:
    print(e)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Now in order to read in pandas dataframe we need to know table name
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(f"Table Name : {cursor.fetchall()}")

df = pd.read_sql_query('SELECT * FROM studies', conn)
print(df)


df_trials = pd.read_sql_query('SELECT * FROM trials', conn)
print(df_trials)

df_trial_params = pd.read_sql_query('SELECT * FROM trial_params', conn)
print(df_trial_params)


df_trial_values = pd.read_sql_query('SELECT * FROM trial_values', conn)
print(df_trial_values)

df = df_trial_values.merge(df_trial_params, how='inner', on=['trial_id'])
print(df)

min_v = np.min(df['value'])
max_v = np.max(df['value'])

print("Beste Configuration:")
min_config = df[df['value'] == min_v]
print(min_config)
for idx, row in min_config.iterrows():
    print(row['param_name'], row['param_value'])
print("-----")
print("Schlechteste Configuration:")
max_config = df[df['value'] == max_v]
for idx, row in max_config.iterrows():
    print(row['param_name'], row['param_value'])
print("-----")

conn.close()

