import math

def log_help_ent(num):
    return (num*(math.log(num,2))+((1-num)*math.log((1-num),2)))*-1
print(log_help_ent(0.5))
print(log_help_ent(0.33333333))
print(((3/5)*log_help_ent(0.33333333))+((2/5)*(log_help_ent(0.5))))
print(2/5-(((3/5)*log_help_ent(0.33333333))+((2/5)*(log_help_ent(0.5)))))
print(2/5-(((3/5)*log_help_ent(0.666666666))))
print(((3/5)*log_help_ent(0.33333333))+((2/5)*(log_help_ent(0.5))))
print(2/5-(((2/5)*log_help_ent(0.5))))
print(1/2-(((1)*log_help_ent(1))))