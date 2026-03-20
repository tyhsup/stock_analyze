from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import mysql.connector
import pandas as pd
import requests
import json

# Create your views here.
def studentMethod(request):
    if (request.method == 'POST'):
        reg = request.body.decode()
        reg = json.loads(reg)
        com_method = str(reg['method'])
        if (com_method == 'upload') :
            try:
                name = str(reg['姓名'])
                number = str(reg['學號'])
                interest = str(reg['興趣'])
                connection = mysql.connector.connect(host = 'localhost',
                                            port = '3306',
                                            user = 'root',
                                            password = 'L123422791ty!',
                                            database = 'ooschool')
                cursor = connection.cursor()
                insert_com = 'INSERT INTO `student` (`姓名`, `學號`, `興趣`) values(%s, %s, %s)'
                insert_data = (name, number, interest)
                cursor.execute(insert_com, insert_data)
                connection.commit()
                cursor.close()
                connection.close()
                return HttpResponse('insert complete')
            except:
                return HttpResponse('insert fail')
            
        elif (com_method == 'reverse') :
            rev_data = reversed(reg['word'])
            return HttpResponse(rev_data)
        
        elif (com_method == 'upper'):
            upper_data = reg['word'].upper()
            return HttpResponse(upper_data)
        
        elif (com_method == 'starify'):
            W = reg['word']
            vowel = ['a','e','i','o','u','A','E','I','O','U']
            for i in W :
                if i in vowel:
                    W = W.replace(i, '*')
            return HttpResponse(W)
            
        else:
            return HttpResponse('mothod error')

