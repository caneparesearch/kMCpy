import json
a=json.load(open("test_input_v3.json"))
for i in a:
    if type(a[i]) is str:
        print('kmc_parser.add_argument("'+i+'",default="'+str(a[i])+'",type='+str(type(a[i])).split("'")[1]+")")
    else:
        print('kmc_parser.add_argument("'+i+'",default='+str(a[i])+",type="+str(type(a[i])).split("'")[1]+")")        