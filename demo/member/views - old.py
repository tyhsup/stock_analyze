from django.shortcuts import render
import mysql.connector
import pandas as pd

# Create your views here.
def show_member_info(request):
    connection = mysql.connector.connect(host = 'localhost',
                                port = '3306',
                                user = 'root',
                                password = 'L123422791ty!',
                                database = 'ooschool')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM `student`;')
    records = pd.DataFrame(cursor.fetchall())
    student_table = records.to_dict('list')
    student_name = student_table[1]
    student_age = student_table[2]
    cursor.close()
    connection.close()
    return render(request, 'test.html',{'student_name' : student_name,
                                        'student_age' : student_age})
