from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import mysql.connector
import pandas as pd
import requests
import json

# Create your views here.
def show_member_info(request):
    connection = mysql.connector.connect(host = 'localhost',
                                port = '3306',
                                user = 'root',
                                password = 'L123422791ty!',
                                database = 'ooschool')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM `student`;')
    records = pd.DataFrame(cursor.fetchall(),columns = ['student_ID','name','age'])
    student_table = records.to_dict('list')
    cursor.close()
    connection.close()
    template = loader.get_template('test.html')
    return HttpResponse(template.render(student_table,request))

def receive_Data(request):
    if (request.method == 'POST') :
        reg = request.body.decode()
        #reg = Talkdb_operation().talkdb_insert('777','request.body','12')
        print(type(reg))
        reg = json.loads(reg)
        print(type(reg), reg)
        return HttpResponse('hello' + str(reg['who']))
 #   else :
  #      return HttpResponse('Error method')
   # reg = request.method
   # return JsonResponse({'foo' : reg})
        