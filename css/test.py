import re
import regex

def extract_data(data, **kwargs):

    new_data=regex.findall(r'P<([A-Z]{3}[\w\W]*?)$',data, overlapped=True)
    new_data=new_data[len(new_data)-1]
    
    new_data=re.sub(r'\s+','',new_data)
    new_data=re.sub(r'"','',new_data)
    new_data=re.sub(r'\'','',new_data)

    return new_data

data = '''                                                                                       8T          SMin   V      eL
                                                                                          IUE        o   A                   AO       a
                                                                                   E   mu UE    e  nrmtiLE o AL       T             e
                                                                                                ae 1                                A                         n
                                                                         -o     Eau           M              U           W                                    L          o     P
                                                    uahN   ue                r               ECiL O  A     W                    A                 PLRFNTA wo A      V  OF   A
                         iuiN       JHaHEIarOD:                                                                                                            TN
                        10D ea   O      pt smb DEd R     P<OAS      LUT O    pilyuaea   ndas on 1 MDEite SteA COEK S0 A           TEH          2 E    ol   Atrnr1De   Aoav88002
                                                                             OIuLEl                                  opn
  RAOI                                                                                                                        917573
 PAKISTAN               DATSAR
     sSO                PAKISTANI    UHNAAD ADN
                         1AUGI             1985                                                       435013396757-3
                                              PESHANVAR                                      PAK
                         OISAR              ULMULK
                         I3MAR             2038                                                         PAKSTAN                                                                    
                          IMAR2078                                                                         99991153425                                                    200838476
                         t
 P<KOAISAR<CUHAMAD<ADNAN<<<<<<<<<<<<<<<<
   CP49175735PAK8508018M28031134250135967573S10
P491773PK'''

print(extract_data(data))