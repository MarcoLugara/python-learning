#User puts hours and rate per hours.
hours = input("Enter number of hours: ")
IntHours = int(hours)
print("Hours: ", IntHours)
#print(type(IntHours))

rph = input("Enter the rate per hours: ")
FloatRph = float(rph)
print("Rate per hours: ", FloatRph)

FullPay = -1

#Define the function that computes the pay
def pay(IntHours, FloatRph):
    if IntHours > 40:
        ExtraHours = IntHours - 40
        print("ExtraHours: ", ExtraHours)
        ExtraHoursPay = ExtraHours * FloatRph * 1.5
        FullPay = 40 * FloatRph + ExtraHoursPay
        return FullPay
    elif IntHours > 0:
        FullPay = IntHours * FloatRph
        return FullPay
    else:
        return "error"
print("The pay is: ", pay(IntHours, FloatRph))

