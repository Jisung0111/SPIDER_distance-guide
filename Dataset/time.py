
import timeit
s = '''

import pandas as pd
import numpy as np
import random

a = np.arange(20000)
np.random.shuffle(a)
df = pd.DataFrame(a, columns=["ID"])

b = np.arange(10000)
b = pd.Series(b, name = "cool")
df[df.isin(b)]
'''
b = '''
import pandas as pd
import numpy as np
import random

a = np.arange(20000)
np.random.shuffle(a)
df = pd.DataFrame(a, columns=["ID"])
df.set_index('ID', inplace = False)
b = np.arange(10000)
b = pd.Series(b, name = "cool")
df.loc[b]
'''
c = '''
import pandas as pd
import numpy as np
import random

a = np.arange(20000)
np.random.shuffle(a)
df = pd.DataFrame(a, columns=["ID"])

b = range(10000)
b = pd.Series(b, name = "cool")
# df.set_index('ID', inplace = False)
df.join(b, how = 'inner')
'''

d = '''
import pandas as pd
import numpy as np
import random

a = np.arange(20000)
np.random.shuffle(a)
df = pd.DataFrame(a, columns=["ID"])

b = range(10000)
b = pd.Series(b, name = "ID")
# df.set_index('ID', inplace = False)
df.merge(b, how = 'inner')
'''

# e = '''
# import pandas as pd
# import numpy as np

# a = np.arange(20000)
# np.random.shuffle(a)
# df = pd.DataFrame(a, columns=["ID"])
#
# b = np.arange(10000)
#
# df.loc[b]
# '''

print(timeit.timeit(stmt = s, number= 10000), "Using isin")
print(timeit.timeit(stmt = b, number= 10000), "Using loc")
print(timeit.timeit(stmt = c, number= 10000), "Using join")
print(timeit.timeit(stmt = d, number= 10000), "Using merge")



