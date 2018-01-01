from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('sample.html')

@app.route('/getdelay',methods=['POST','GET'])

def get_delay():
    if request.method=='POST':
        result=request.form['title']
        val = []
        val.append(result)
        result = np.array(val)
        #
        pkl_vect = open('train_vectorizer.pkl', 'rb')
        pkl_mat = open('train_matrix.pkl', 'rb')
        pkl_label = open('train_result.pkl', 'rb')
        #
        vect_model = pickle.load(pkl_vect, encoding='latin1')
        mat_model = pickle.load(pkl_mat, encoding='latin1')
        label_model = pickle.load(pkl_label, encoding='latin1')
        #
        test_vect = vect_model.transform(result)
        csr_sim = cosine_similarity(test_vect,mat_model)
        output = []
        for row in csr_sim:
            #get single row from input scr matrix
            row = np.array(row)
            #sort and get 'k' nearest neighbor indexes
            max_row = np.argpartition(-row, 2)
            top_rows = max_row[:2]
            result = []
            for i in range(len(top_rows)):
                val = top_rows[i]

                result.append(label_model[val])
            
            output.append(result)
        
    
        #
        
        
        return render_template('sample.html',prediction=output)

    
if __name__ == '__main__':
	app.run()




