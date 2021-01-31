import turtle
import time


def drawGap():
    turtle.penup()
    turtle.fd(5)


def drawLine(draw):
    drawGap()
    turtle.pendown() if draw else turtle.penup()
    turtle.fd(40)
    drawGap()
    turtle.right(90)


def drawDigit(digit):
    drawLine(1) if digit in [2, 3, 4, 5, 6, 8, 9] else drawLine(0)
    drawLine(1) if digit in [0, 1, 3, 4, 5, 6, 7, 8, 9] else drawLine(0)
    drawLine(1) if digit in [0, 2, 3, 5, 6, 8, 9] else drawLine(0)
    drawLine(1) if digit in [0, 2, 6, 8] else drawLine(0)
    turtle.left(90)
    drawLine(1) if digit in [0, 4, 5, 6, 8, 9] else drawLine(0)
    drawLine(1) if digit in [0, 2, 3, 5, 6, 7, 8, 9] else drawLine(0)
    drawLine(1) if digit in [0, 1, 2, 3, 4, 7, 8, 9] else drawLine(0)
    turtle.left(180)
    turtle.penup()
    turtle.fd(20)


def drawDate(date):
    for i in date:
        if i == '-':
            turtle.write("年", font=("Arial", 18, "normal"))
            turtle.fd(40)
        elif i == '=':
            turtle.write("月", font=("Arial", 18, "normal"))
            turtle.fd(40)
        elif i == '+':
            turtle.write("日", font=("Arial", 18, "normal"))
            turtle.fd(40)
        else:
            drawDigit(eval(i))


def main():
    turtle.setup(800, 350, 200, 200)
    turtle.penup()
    turtle.fd(-300)
    turtle.pensize(5)
    turtle.pencolor("red")
    curtime = time.gmtime()
    tim = time.strftime("%Y-%m=%d+", curtime)
    drawDate(tim)
    turtle.hideturtle()
    turtle.done()


main()
