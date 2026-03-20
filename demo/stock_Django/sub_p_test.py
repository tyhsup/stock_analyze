import subprocess
import mySQL_OP

class sub_Pro_com:
    
    def load_stock_number(self,table):
        SQL_OP = mySQL_OP.OP_Fun()
        dl_cost_table = SQL_OP.sel_table_data(table_name = table)
        stock_number = dl_cost_table['number'].drop_duplicates().to_list()
        return stock_number

sub_process = sub_Pro_com()
stock_list = sub_process.load_stock_number('stock_cost')

for i in range(len(stock_list)):
    test = subprocess.run(['python', 'model_train.py', 'stock_cost', str(stock_list[i])], stdin = subprocess.PIPE,
                      capture_output = True, text = True, check = True)
    print(test.stdout)

