import joblib
import numpy as np
import pandas as pd
import json
import ast


# 저장된 모델 불러오기
loaded_knn = joblib.load('knn_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')
filtered_df = pd.read_csv('train_user.csv', index_col=0)


class EASE:
    def __init__(self, reg_lambda=500):
        self.reg_lambda = reg_lambda
        self.item_weights = None

    def fit(self, X):
        G = np.dot(X.T, X) + np.eye(X.shape[1]) * self.reg_lambda

        P = np.linalg.inv(G)
        B = P / -np.diag(P)
        np.fill_diagonal(B, 0)
        self.item_weights = B

    def predict(self, X):
        return np.dot(X, self.item_weights)
    
# 테스트 데이터
def handler(event, context):


        data = json.loads(event['body'])
        values = pd.DataFrame([data])
        value = values.iloc[:,:-1] # 마지막 칼럼은 맞춘 문제 번호들
        
        scaled_test = loaded_scaler.transform(value)
        distances, indices = loaded_knn.kneighbors([scaled_test[0]])
        df = filtered_df.iloc[indices[0]]
        df.reset_index(drop=True, inplace=True)
        df['solvedProblemList'] = df['solvedProblemList'].apply(ast.literal_eval)
        all_problems = set()
        for row in df.itertuples():
            all_problems.update(row.solvedProblemList)
        all_problems = sorted(all_problems)

        interaction_matrix = pd.DataFrame(0, index=df['id'], columns=all_problems)

        for row in df.itertuples():
            interaction_matrix.loc[row.id, row.solvedProblemList] = 1

        model = EASE(reg_lambda=100)
        model.fit(interaction_matrix)
        X_test = interaction_matrix[0:1]
        predicted_scores = model.predict(X_test)
        column_names = interaction_matrix.columns
        score_mapping = {column_names[i]: score for i, score in enumerate(predicted_scores[0])}

        solved_problems = values['solvedProblemList'][0]

        filtered_scores = {problem: score for problem, score in score_mapping.items() if problem not in solved_problems}

        sorted_problems = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)

        top_15_problems = sorted_problems[:15]

        top_15_problem_numbers = [problem[0] for problem in top_15_problems]
        top_15_problem_numbers = list(map(int, top_15_problem_numbers))


        # 결과 반환
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'result': top_15_problem_numbers})
        }
    
