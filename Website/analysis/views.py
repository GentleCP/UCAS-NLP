from django.shortcuts import render
from django.http import HttpResponse

from NaiveBayes import naive_bayes
from NaiveBayes.naive_bayes import NaiveBayes


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
        # s = 'success'
        # r_set.append({'result': r, 'status': s, 'model': 'svm'})

        # if request.POST['type'] == 'dl':
        #     print('dl')
        r = 'xxxxxx'
        s = 'success'
        r_set.append({'result': r, 'status': s, 'model': 'dl'})

        # if request.POST['type'] == 'nb':
        #     print('nb')
        nb = naive_bayes.load_model('../../NaiveBayes/bayes_model.pkl')
        r = nb.predict(data)
        s = 'success'
        r_set.append({'result': r, 'status': s, 'model': 'nb'})

        # if request.POST['type'] == 'lstm':
        #     print('lstm')
        r = 'xxxxxx'
        s = 'success'
        r_set.append({'result': r, 'status': s, 'model': 'lstm'})


        # do something

        #return render(request, 'index.html', {'result': r, 'status': s, 'data': data})
        return render(request, 'index.html', {'r_set': r_set, 'data': data})

    else:
        return HttpResponse('your method is not valid!')

