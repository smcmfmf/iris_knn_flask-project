import pickle
import numpy as np
from flask import Flask, request, render_template, url_for

app = Flask(__name__)

try:
    with open('model/iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: 'model/iris_model.pkl' 파일을 찾을 수 없습니다.")
    print("먼저 제공해주신 첫 번째 Python 스크립트를 실행하여 모델을 생성하세요.")
    exit()

# 2. 붓꽃 정보에 'description' (설명) 추가
iris_info = {
    0: {
        'name': 'Setosa (부채붓꽃)',
        'img_path': 'setosa.jpg',
        'description': '세 종 중에서 가장 작은 편이며, 꽃잎과 꽃받침이 확연히 구분됩니다. 주로 습지 근처에서 발견됩니다.'
    },
    1: {
        'name': 'Versicolor (푸른붓꽃)',
        'img_path': 'versicolor.jpg',
        'description': '이름처럼 다양한 파란색, 보라색 꽃을 피웁니다. Setosa와 Virginica의 중간 정도 특징을 가집니다.'
    },
    2: {
        'name': 'Virginica (버지니카붓꽃)',
        'img_path': 'virginica.jpg',
        'description': '세 종 중에서 가장 큰 꽃을 피우는 경향이 있으며, 짙은 보라색 꽃잎이 특징입니다.'
    }
}


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_name = None
    prediction_img_url = None
    prediction_desc = None  # 설명을 담을 변수 추가

    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])

            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

            prediction_index = model.predict(features)[0] 
            
            # 딕셔너리에서 모든 정보(이름, 이미지, 설명)를 가져옴
            info = iris_info[prediction_index]
            prediction_name = info['name']
            prediction_img_url = url_for('static', filename=info['img_path'])
            prediction_desc = info['description'] # 설명 변수에 할당

        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            prediction_name = "오류 발생"

    # HTML 템플릿에 'description' 변수도 함께 전달
    return render_template('index.html', 
                           prediction=prediction_name, 
                           image_url=prediction_img_url,
                           description=prediction_desc)

# 4. 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)