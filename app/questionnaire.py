#kerasをインポートする
from keras import backend
#from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense,Activation,LSTM
from keras.utils import np_utils
from keras.models import load_model    
from keras.layers.core import Dropout

#pandasとnumpyをインポートする
import pandas as pd
import numpy as np

#すべての警告を非表示にする
#import warnings
#warnings.filterwarnings('ignore')

#Word2Vecをインポートする
from gensim.models.word2vec import Word2Vec

#MeCab:形態素解析器のインポート
import MeCab

class LstmClass:
    def __init__(self):
        #中間総数
        self.n_hidden = 32
        #入力データ数
        self.data_dim = 50
        self.timesteps = 30
        #出力データ数
        self.num_classes = 6
        
    def lstmModel(self):
        #LSTMの設定
        lstm_model = Sequential()
        lstm_model.add(LSTM(self.n_hidden, return_sequences=True,
        input_shape=(self.timesteps, self.data_dim)))  # 32次元のベクトルのsequenceを出力する

        lstm_model.add(LSTM(self.n_hidden))  # 32次元のベクトルを一つ出力する
        lstm_model.add(Dense(self.num_classes, activation='softmax'))

        lstm_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        return lstm_model

    def morphologicAlanalysis (self,input_text):
        words_arr2 = []
        
        for suvey_row in range(1):
            #input_text = input()
            text = input_text
            #形態素解析の実行
            tagger = MeCab.Tagger()
            words = tagger.parse(text).splitlines()

            words_arr = []
            for i in range(self.timesteps):
                if(i < len(words)):
                    word_tmp = words[i].split()[0]
                    words_arr.append(word_tmp)
                else:
                    word_tmp = 'A'
                    words_arr.append(word_tmp)
        
        return words_arr

    def lstmProb(self,lstm_model,words_arr):
        #LSTMの重さをロード
        lstm_model.load_weights('app/w2v_lstm_weights.h5')
        #テストデータの予測を行う
        model_path = 'app/word2vec.gensim.model'
        w2v_model = Word2Vec.load(model_path)
        #予測を行う
        x_test =[]
        x_test.append(w2v_model[words_arr])
        x_test = np.array(x_test).astype(np.float32)
        prob = lstm_model.predict(x_test)
        
        return prob
        
    def resultView(self,prob):
        #結果を表示する
        return "このアンケートの顧客満足度は" + str(max(prob[0])) + 'の確率で' + str(np.argmax(prob[0])) + "です。"

def questionnaire_result(input_text):
    #LSTMのモデルを決定
    lstm = LstmClass()
    #LSTMによる予測を行う
    prob = lstm.lstmProb(lstm.lstmModel(),lstm.morphologicAlanalysis(input_text))
    #結果を表示する
    result = lstm.resultView(prob)

    return result

'''
def main():
    #LSTMのモデルを決定
    lstm = LstmClass()
    #LSTMによる予測を行う
    prob = lstm.lstmProb(lstm.lstmModel(),lstm.morphologicAlanalysis())
    #結果を表示する
    lstm.resultView(prob)

if __name__ == '__main__':
    main()
'''

'''
#kerasをインポートする
from keras import backend
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense,Activation,LSTM
from keras.utils import np_utils
from keras.models import load_model    
from keras.layers.core import Dropout

#pandasとnumpyをインポートする
import pandas as pd
import numpy as np

#すべての警告を非表示にする
import warnings
warnings.filterwarnings('ignore')

#Word2Vecをインポートする
from gensim.models.word2vec import Word2Vec

#MeCab:形態素解析器のインポート
import MeCab

class LstmLearnClass:
    def __init__(self):
        #中間総数
        self.n_hidden = 32
        #入力データ数
        self.data_dim = 50
        self.timesteps = 30
        #学習回数
        self.epochs = 1000   
        #出力データ数
        self.num_classes = 6
        
    def dataInput(self):
        #csv1ファイルを読み取る
        survey = pd.read_csv("survey.csv")
        #欠損値を削除
        survey = survey.dropna()
        survey.isna().sum()
        
        return survey
    
    def dataXtrain(self,survey):
        #Word2Vecで使用するデータのパス指定
        model_path = './word2vec.gensim.model'
        w2v_model = Word2Vec.load(model_path)
        #単語をベクトルに変換
        x_train =[]
        for survey_row in survey:
            x_train.append(w2v_model[survey_row])
        x_train = np.array(x_train)
        
        return x_train
        
    def dataYtrain(self,survey):
        #教師データを設定
        y_train = survey['satisfaction']
        y_train = np_utils.to_categorical(y_train)
        
        return y_train
        
    def lstmModel(self):
        #LSTMの設定
        lstm_model = Sequential()
        lstm_model.add(LSTM(self.n_hidden, return_sequences=True,
        input_shape=(self.timesteps, self.data_dim)))  # 32次元のベクトルのsequenceを出力する
        lstm_model.add(LSTM(self.n_hidden))  # 32次元のベクトルを一つ出力する
        lstm_model.add(Dense(self.num_classes, activation='softmax'))

        lstm_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        return lstm_model

    def morphologicAlanalysis(self,survey):
        #形態素解析の実行
        words_arr2 = []
        for suvey_row in range(len(survey)):
            text = survey["comment"].iloc[suvey_row]
            tagger = MeCab.Tagger()
            words = tagger.parse(text).splitlines()

            words_arr = []
            for i in range(self.timesteps):
                if(i < len(words)):
                    word_tmp = words[i].split()[0]
                    words_arr.append(word_tmp)
                else:
                    #入力データの余った配列に文字列を設定する
                    word_tmp = 'A'
                    words_arr.append(word_tmp)
            words_arr2.append(words_arr)
        return words_arr2

    def lstmLearn(self,lstm_model,x_train,y_train):
        #LSTMの重さ更新
        lstm_model.fit(x_train, y_train,batch_size=64, epochs=self.epochs)
        #LSTMの重さ保存
        lstm_model.save_weights('w2v_lstm_weights.h5')

def main():
    #LSTM実行
    lstm = LstmLearnClass()
    survey = lstm.dataInput()
    x_train = lstm.morphologicAlanalysis(survey)
    X = lstm.dataXtrain(x_train)
    Y = lstm.dataYtrain(survey)
    #LSTMの重さ保存
    lstm.lstmLearn(lstm.lstmModel(),X,Y)

if __name__ == '__main__':
    main()
'''