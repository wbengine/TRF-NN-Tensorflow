from libcpp cimport bool

cdef extern from 'wb-lhash.h' namespace 'wb':
    bool LHash_IncSort[KeyT](KeyT k1, KeyT k2)

cdef extern from 'wb-trie.h' namespace 'wb':

    cdef cppclass Trie[KeyT, DataT]:
        Trie()
        void Release()
        size_t TotalMemCost()
        void Clean()
        void Fill(DataT d)
        DataT* GetData()
        bool IsDataLegal()
        DataT* Find(const KeyT *p_pIndex, int nIndexLen, bool &bFound)
        DataT* Insert(const KeyT *p_pIndex, int nIndexLen, bool &bFound)
        void Remove(const KeyT *p_pIndex, int nIndexLen)
        Trie[KeyT, DataT] *FindTrie(const KeyT *p_pIndex, int nIndexLen, bool &bFound)
        Trie[KeyT, DataT] *InsertTrie(const KeyT *p_pIndex, int nIndexLen, bool &bFound)
        void FindParallel(KeyT *pIndexBuf, int *pIndexLen, int index_stride, int index_num, DataT *pResults, DataT none_value, int thread_num)

    cdef cppclass TrieIter[KeyT, DataT]:
        TrieIter(Trie[KeyT, DataT] *ptrie, bool(*sort)(KeyT, KeyT))
        void Init()
        Trie[KeyT, DataT] *Next(KeyT &key)

    cdef cppclass TrieIter1[KeyT, DataT]:
        TrieIter1(Trie[KeyT, DataT] *ptrie, KeyT *pIndex, int nLevel, bool(*sort)(KeyT, KeyT))
        void Init()
        Trie[KeyT, DataT] *Next(int &nLen)

    cdef cppclass TrieIter2[KeyT, DataT]:
        TrieIter2(Trie[KeyT, DataT] *ptrie, KeyT *pIndex, int level, bool(*sort)(KeyT, KeyT))
        void Init()
        Trie[KeyT, DataT] *Next()
