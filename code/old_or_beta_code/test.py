import datetime

startTime = datetime.datetime.now()

while True:
    currentTime = datetime.datetime.now()
    timeDifference = (currentTime - startTime).total_seconds()

    if timeDifference > 5:
        print('Es ist nun 5 Sekunden später:   ', currentTime.second)
        startTime = currentTime