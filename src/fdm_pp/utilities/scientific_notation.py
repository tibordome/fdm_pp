#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 12:05:30 2021

@author: tibor
"""

def eTo10(st):
    """Replace e+{xy} by "10^{xy}" etc..
    Note that "st" should be handed over with {:.2E}.
    Works for exponents 00 to 19, plus and minus, e and E"""
    st_l = list(st)
    i = 0
    drop_last_char = True
    while i < len(st_l):
        if st_l[i] == "e" or st_l[i] == "E":
            if st_l[i+2] == "0" and st_l[i+3] == "0":
                if st_l[i] == "e":
                    split_string = st.split("e", 1)
                else:
                    split_string = st.split("E", 1)
                return split_string[0]
            if st_l[i+1] == "+":
                st_l[i] = "1"
                st_l[i+1] = "0"
                if st_l[i+2] == "0" and st_l[i+3] == "1":
                    st_l[i+2] = "\u00B9"
                elif st_l[i+2] == "0" and st_l[i+3] == "2":
                    st_l[i+2] = "\u00b2"
                elif st_l[i+2] == "0" and st_l[i+3] == "3":
                    st_l[i+2] = "\u00B3"
                elif st_l[i+2] == "0" and st_l[i+3] == "4":
                    st_l[i+2] = "\u2074"
                elif st_l[i+2] == "0" and st_l[i+3] == "5":
                    st_l[i+2] = "\u2075"
                elif st_l[i+2] == "0" and st_l[i+3] == "6":
                    st_l[i+2] = "\u2076"
                elif st_l[i+2] == "0" and st_l[i+3] == "7":
                    st_l[i+2] = "\u2077"
                elif st_l[i+2] == "0" and st_l[i+3] == "8":
                    st_l[i+2] = "\u2078"
                elif st_l[i+2] == "0" and st_l[i+3] == "9":
                    st_l[i+2] = "\u2079"
                elif st_l[i+2] == "1" and st_l[i+3] == "0":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2070"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "1":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u00B9"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "2":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u00b2"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "3":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u00B3"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "4":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2074"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "5":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2075"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "6":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2076"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "7":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2077"
                    drop_last_char = False
                elif st_l[i+2] == "1" and st_l[i+3] == "8":
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2078"
                    drop_last_char = False
                else:
                    assert st_l[i+2] == "1" and st_l[i+3] == "9"
                    st_l[i+2] = "\u00B9"
                    st_l[i+3] = "\u2079"
                    drop_last_char = False
                    
                st_l.insert(i, "\u2800") # Add blank character
                st_l.insert(i+1, "\u2715") # Add \times character
                st_l.insert(i+2, "\u2800") # Add blank character
                if drop_last_char == True:
                    st_l = st_l[:-1]
            else:
                assert st_l[i+1] == "-"
                st_l[i] = "1"
                st_l[i+1] = "0"
                    
                if st_l[i+2] == "0" and st_l[i+3] == "1":
                    st_l[i+3] = "\u00B9"
                elif st_l[i+2] == "0" and st_l[i+3] == "2":
                    st_l[i+3] = "\u00b2"
                elif st_l[i+2] == "0" and st_l[i+3] == "3":
                    st_l[i+3] = "\u00B3"
                elif st_l[i+2] == "0" and st_l[i+3] == "4":
                    st_l[i+3] = "\u2074"
                elif st_l[i+2] == "0" and st_l[i+3] == "5":
                    st_l[i+3] = "\u2075"
                elif st_l[i+2] == "0" and st_l[i+3] == "6":
                    st_l[i+3] = "\u2076"
                elif st_l[i+2] == "0" and st_l[i+3] == "7":
                    st_l[i+3] = "\u2077"
                elif st_l[i+2] == "0" and st_l[i+3] == "8":
                    st_l[i+3] = "\u2078"
                elif st_l[i+2] == "0" and st_l[i+3] == "9":
                    st_l[i+3] = "\u2079"
                elif st_l[i+2] == "1" and st_l[i+3] == "0":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2070"
                elif st_l[i+2] == "1" and st_l[i+3] == "1":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u00B9"
                elif st_l[i+2] == "1" and st_l[i+3] == "2":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u00b2"
                elif st_l[i+2] == "1" and st_l[i+3] == "3":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u00B3"
                elif st_l[i+2] == "1" and st_l[i+3] == "4":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2074"
                elif st_l[i+2] == "1" and st_l[i+3] == "5":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2075"
                elif st_l[i+2] == "1" and st_l[i+3] == "6":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2076"
                elif st_l[i+2] == "1" and st_l[i+3] == "7":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2077"
                elif st_l[i+2] == "1" and st_l[i+3] == "8":
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2078"
                else:
                    assert st_l[i+2] == "1" and st_l[i+3] == "9"
                    st_l[i+3] = "\u00B9"
                    st_l += "\u2079"
                
                st_l[i+2] = "\u207b" # Minus in the exponent
                st_l.insert(i, "\u2800") # Add blank character space
                st_l.insert(i+1, "\u2715") # Add \times character
                st_l.insert(i+2, "\u2800") # Add blank character space
        i += 1
    if st_l[0] == "1" and st_l[1] == "\u002E" and st_l[2] == "0" and st_l[3] == "0": # Remove 1.00 x 
        return "".join(st_l[7:])
    return "".join(st_l)

"""
area = 2
print("The area of your rectangle is {} cm\u00b2".format(area))
test = "2.88E-19"
print("The string", test, "becomes", eTo10(test))"""
