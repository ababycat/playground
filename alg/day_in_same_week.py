
# 判断两天是否在同一周


def confirm_day_in_same_week(y1, m1, d1, y2, m2, d2):
    w1 = week(y1, m1, d1)
    w2 = week(y2, m2, d2)

    d1 = day(y1, m1, d1)
    d2 = day(y2, m2, d2)

    if d1 - d2 > 0:
        d1, d2 = d2, d1
        w1, w2 = w2, w1

    if d2 - d1 > 7:
        return False
    
    w1 += 1
    w2 += 1

    if w2 < w1:
        return False

    return True
