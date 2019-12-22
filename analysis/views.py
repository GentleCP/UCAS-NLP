from django.shortcuts import render
from django.http import HttpResponse

from NaiveBayes import naive_bayes
from lstm.predict import predict as lstm_predict

from CNN.cnn import cnn_predict
# Create your views here.


def index(request):
    if request.method == 'GET':
        return render(request, 'index.html')

    elif request.method == 'POST':

        data = request.POST['data']
        r = ''
        s = ''
        r_set = []
        # if request.POST['type'] == 'svm':
        #     print('svm')
            # do something
        # r = 'xxxxxx'
        # r_set.append({'result': r, 'model': 'svm'})

        if request.POST['type'] == 'dl':
            print('dl')
            r = 'xxxxxx'
            # r = cnn_predict(data)
            r_set.append({'result': r, 'model': 'cnn'})

        if request.POST['type'] == 'nb':
            print('nb')
            nb = naive_bayes.load_model('NaiveBayes/bayes_model.pkl')
            r = nb.predict(data)
            r_set.append({'result': r, 'model': 'nb'})

        if request.POST['type'] == 'lstm':
            print('lstm')
            r = lstm_predict(data)
            r_set.append({'result': r, 'model': 'lstm'})


        # do something

        #return render(request, 'index.html', {'result': r, 'status': s, 'data': data})
        return render(request, 'index.html', {'r_set': r_set, 'data': data})

    else:
        return HttpResponse('your method is not valid!')

