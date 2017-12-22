cimport ctrie
from cpython cimport array
import array
from libcpp cimport bool

ctypedef int KeyT
ctypedef int DataT
ctypedef ctrie.Trie[KeyT, DataT] cTrie
ctypedef ctrie.TrieIter[KeyT, DataT] cIter
ctypedef ctrie.TrieIter1[KeyT, DataT] cIter1
ctypedef ctrie.TrieIter2[KeyT, DataT] cIter2

def trie():
    return Trie().create()

cdef trie_warper(cTrie *p):
    t = Trie()
    t._c_trie = p
    return t

cdef class Trie:
    """
    A trie class. Usage: <python code>
    
    import trie
    t = trie.trie()
    t.insert([1, 2, 3], 10.0)
    t.insert([1, 2], 5.0)
    print(t.find([1, 2, 3])
    print(t.find([1, 2])
    print([1, 2] in t) 
    
    """
    cdef cTrie *_c_trie
    cdef bool _need_release

    def __cinit__(self):
        self._c_trie = NULL
        self._need_release = 0

    def __dealloc__(self):
        if self._need_release:
            del self._c_trie

    def create(self):
        if self._c_trie is NULL:
            self._c_trie = new cTrie()
            self._need_release = 1
        else:
            raise MemoryError('Tire has been inilized!')
        return self

    def clean(self):
        """clean the trie"""
        self._c_trie.Clean()
        
    def insert(self, key_list, value=None):
        """
        insert a value.
        If value is not None, then insert the value;
        otherwise, just create a sub-trie
        Return a sub trie
        """
        cdef array.array keys = array.array('i', key_list)
        cdef bool bFound = False
        cdef cTrie *psub = self._c_trie.InsertTrie(keys.data.as_ints, len(key_list), bFound)
        if value is not None:
            psub.GetData()[0] = value
        return trie_warper(psub)
        
    def setdefault(self, key_list, value):
        """
        Set the default value.
        If the trie exits, this method return the sub-trie.
        If not, this method insert the value and return the sub-trie
        """
        cdef array.array keys = array.array('i', key_list)
        cdef bool bFound = False
        cdef cTrie *psub = self._c_trie.InsertTrie(keys.data.as_ints, len(key_list), bFound)
        if not bFound or not psub.IsDataLegal():
            psub.GetData()[0] = value
        return trie_warper(psub)
        
    def find(self, key_list):
        cdef array.array keys = array.array('i', key_list)
        cdef bool bFound = False
        cdef DataT *pData = self._c_trie.Find(keys.data.as_ints, len(key_list), bFound)
        if bFound == False:
            return None
        return pData[0]

    def remove(self, key_list):
        cdef array.array keys = array.array('i', key_list)
        self._c_trie.Remove(keys.data.as_ints, len(key_list))

    def find_trie(self, key_list):
        cdef array.array keys = array.array('i', key_list)
        cdef bool bFound = False
        cdef cTrie *psub = self._c_trie.FindTrie(keys.data.as_ints, len(key_list), bFound)
        if bFound == False:
            return None
        return trie_warper(psub)

    def find_parallel(self, keys):
        key_num = len(keys)
        key_len = [len(k) for k in keys]
        stride = max(key_len)
        key_merge = []
        for k in keys:
            key_merge += k + [0] * (stride - len(k))

        cdef array.array index_buf = array.array('i', key_merge)
        cdef array.array index_len = array.array('i', key_len)
        cdef array.array res = array.array('i', [0]*key_num)
        self._c_trie.FindParallel(index_buf.data.as_ints, index_len.data.as_ints,
                                  stride, key_num,
                                  res.data.as_ints, -1,
                                  4)
        return list(res)

    def __contains__(self, key_list):
        cdef array.array keys = array.array('i', key_list)
        cdef bool bFound = False
        self._c_trie.Find(keys.data.as_ints, len(key_list), bFound)
        return bFound

    @property
    def data(self):
        return self._c_trie.GetData()[0]

    @data.setter
    def data(self, value):
        self._c_trie.GetData()[0] = value


cdef class SubIter:
    """
    traverse the the  child of current node
    """
    cdef cIter *_c_iter

    def __cinit__(self, Trie t, bool is_sorted=False):
        if is_sorted:
            self._c_iter = new cIter(t._c_trie, ctrie.LHash_IncSort[KeyT])
        else:
            self._c_iter = new cIter(t._c_trie, NULL)

    def __dealloc__(self):
        del self._c_iter

    def init(self):
        self._c_iter.Init()

    def __next__(self):
        cdef KeyT key = 0
        cdef cTrie *p = self._c_iter.Next(key)
        if p == NULL:
            raise StopIteration
        return key, p.GetData()[0]

    def __iter__(self):
        return self


cdef class TrieIter:
    """
    traverse all the node in the whole trie
    """
    cdef cIter1 *_c_iter
    cdef array.array _context

    def __cinit__(self, Trie t, bool is_sorted=False, int max_key_len = 100):
        self._context = array.array('i', [0]*max_key_len)
        if is_sorted:
            self._c_iter = new cIter1(t._c_trie, self._context.data.as_ints, 1, ctrie.LHash_IncSort[KeyT])
        else:
            self._c_iter = new cIter1(t._c_trie, self._context.data.as_ints, 1, NULL)

    def __dealloc__(self):
        del self._c_iter

    def init(self):
        self._c_iter.Init()
        
    def __next__(self):
        cdef int n = 0
        cdef cTrie *p = self._c_iter.Next(n)
        if p == NULL:
            raise StopIteration
        return self._context[0:n].tolist(), p.GetData()[0]

    def __iter__(self):
        return self


cdef class LevelIter:
    """
    traverse all the node at a fix level
    """
    cdef cIter2 *_c_iter
    cdef array.array _context

    def __cinit__(self, Trie t, int level, bool is_sorted=False):
        self._context = array.array('i', [0]*level)
        if is_sorted:
            self._c_iter = new cIter2(t._c_trie, self._context.data.as_ints, level, ctrie.LHash_IncSort[KeyT])
        else:
            self._c_iter = new cIter2(t._c_trie, self._context.data.as_ints, level, NULL)

    def __dealloc__(self):
        del self._c_iter

    def init(self):
        self._c_iter.Init()

    def __next__(self):
        cdef cTrie *p = self._c_iter.Next()
        if p == NULL:
            raise StopIteration
        return self._context.tolist(), p.GetData()[0]

    def __iter__(self):
        return self

