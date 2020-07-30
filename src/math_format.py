import string
import decimal

class MathTextSciFormatter(string.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)


def format_decimal(x, prec=2):
    tup = x.as_tuple()
    digits = list(tup.digits[:prec + 1])
    sign = '-' if tup.sign else ''
    dec = ''.join(str(i) for i in digits[1:])
    exp = x.adjusted()
    return '{sign}{int}.{dec}e{exp}'.format(sign=sign, int=digits[0], dec=dec, exp=exp)
    
def format_decimal1(x, prec=2):
    decimal_number = decimal.Decimal(x)
    tup = decimal_number.as_tuple()
    digits = list(tup.digits[:prec + 1])
    sign = '-' if tup.sign else ''
    dec = ''.join(str(i) for i in digits[1:])
    exp = decimal_number.adjusted()
    return '{sign}{int}.{dec}e{exp}'.format(sign=sign, int=digits[0], dec=dec, exp=exp)
    
def sci_str(dec):
    return ('{:.' + str(len(dec.normalize().as_tuple().digits) - 1) + 'E}').format(dec)

def format_decimal2(x, prec=2):
    decimal_number = decimal.Decimal(x)
    tup = decimal_number.as_tuple()
    digits = list(tup.digits[:prec + 1])
    sign = '-' if tup.sign else ''
    dec = ''.join(str(i) for i in digits[1:])
    exp = decimal_number.adjusted()
    return '{sign}{int}.{dec}e{exp}'.format(sign=sign, int=digits[0], dec=dec, exp=exp)
    
def format_number(x, fmt='%1.2e'):
    s = fmt % x
    decimal_point = '.'
    positive_sign = '+'
    tup = s.split('e')
    significand = tup[0].rstrip(decimal_point)
    sign = tup[1][0].replace(positive_sign, '')
    exponent = tup[1][1:].lstrip('0')
    if exponent:
        exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
    return "${}$".format(s)
    
                
#str1 = str(0.09)

#my_number = 0.09
#print format_decimal1(my_number)

#r = 0.00000095

#print '%1.3e' % r 

#print format_number(r, fmt='%1.2e')

