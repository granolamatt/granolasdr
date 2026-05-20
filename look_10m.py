import json, sys
count = 0
for line in open(sys.argv[1]):
    try:  
        d = json.loads(line.strip())
        if 28000000 <= d.get('freq', 0) <= 29000000:
            print(d)
            count += 1 
            if count >= 10: break
    except: pass
if not count: print('No 10m signals found in log')

