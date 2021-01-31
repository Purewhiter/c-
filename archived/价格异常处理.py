while 1:
    try:
        count = int(input("input numbers:"))
        price = int(input("input price per item:"))
        pay = count*price
        print("total price is", pay)
        break
    except:
        print("something went wrong,try again")
