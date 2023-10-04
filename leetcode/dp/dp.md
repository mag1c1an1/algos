# dp

198
70
746
2466
213

494
322
416
279
518

1143
72
583
712
1458
97

## group one

### longest-increasing-subsequence

```c++
//dfs
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n =nums.size();
        int ans = 1;
        unordered_map<int,int> mm;
        for(int x  = 0;x < n ;x ++ ) {
           ans = max(dfs(nums,x,mm),ans);
        }
        return ans;
    }
    auto dfs(vector<int>& nums, int i,unordered_map<int, int> &mm) -> int {
        if(i < 0) return 0;
        if(i == 0) return 1;
        if(mm.count(i) > 0) return mm[i];
        int n = nums.size();
        int ans = 0;
        for(int j = 0; j < i;j ++) {
            if(nums[j]  < nums[i])
            {
                ans   = max(dfs(nums,j,mm),ans);
            }
        }
        ans++;
        mm[i] = ans;
        return ans;
    }
};

// dp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n =nums.size();
        auto f = vector<int>(n,1);
        int ans = f[0];
        for(int i  = 0;i < n ;i++ ) {
           for(int j = 0;j<i;j++){
               if(nums[j] < nums[i]) {
                   f[i] = max(f[i],f[j]+1);
               }
           } 
        }
        for(auto v : f) 
        {
            if(v > ans) {
                ans = v;
            }
        }
        return ans;
    }
};

// greedy
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int> g;
        for(int x :nums)
        {
            auto it = lower_bound(g.begin(), g.end(), x);
            if(it == g.end()) g.push_back(x);
            else *it = x;
        }
        return g.size();
    }
}

//
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        auto end = nums.begin();
        for(int x :nums)
        {
            auto it = lower_bound(nums.begin(),end,x);
            *it = x;
            if(it == end) end++;
        }
        return end-nums.begin(); 
    }
};
```

### number-of-longest-increasing-subsequence

```c++
// dp
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size(),maxLen = 0,ans = 0;
        vector<int> f(n),cnt(n);
        for(int i = 0; i < n;++i) {
           f[i]  = 1, cnt[i] = 1;
           for(int j = 0;j<i;++j) 
           {
               if (nums[j]<nums[i]) {
                   if(f[j]+ 1 > f[i]){
                       f[i] = f[j]+1;
                    cnt[i]  = cnt[j];
                   }else if(f[j] + 1 == f[i]) {
                       cnt[i] += cnt[j];
                   }
               }
           }
           if(f[i] > maxLen) {
               maxLen = f[i];
               ans = cnt[i];
           }else if(f[i] == maxLen)
           {
               ans += cnt[i];
           }
        }
        return ans;
    }
};
// binary
class Solution {
int bs(int n, function<bool(int)> f){
    int l = 0, r= n;
    while( l < r) {
        int mid = (l+r)/2;
        if(f(mid)) {
            r = mid;
        } else {
            l = mid  +1;
        }
    }
    return l;
}

public:
    int findNumberOfLIS(vector<int>& nums) {
        vector<vector<int>> d, cnt;
        for(auto v  : nums ){ 
            int i  = bs(d.size(),[&](int i) {return d[i].back() >= v;});
            int c = 1;
            if(i>0) {
               int k = bs(d[i-1].size(),[&](int k) {return d[i-1][k] < v;}); 
               c = cnt[i-1].back() - cnt[i-1][k];
            }
            if( i == d.size()) {
                d.push_back({v});
                cnt.push_back({0,c});
            }else {
                d[i].push_back(v);
                cnt[i].push_back(cnt[i].back()+c);
            }
        }
    return cnt.back().back();
    }
};
```

### find-the-longest-valid-obstacle-course-at-each-position/

```c++
// greedy + binary
class Solution {
public:
    vector<int> longestObstacleCourseAtEachPosition(vector<int>& obstacles) {
        int n = obstacles.size();
        vector<int> g,ans(n); 
        for(int i = 0;i<n;++i){
            auto it = upper_bound(g.begin(), g.end(), obstacles[i]);
            if(it ==  g.end()){
                g.push_back(obstacles[i]);
                ans[i] = g.size();
            }else {
                *it = obstacles[i];
                ans[i] = it-g.begin()+1; 
            }
        }
        return ans;
    }
};
```

### minimum-number-of-removals-to-make-mountain-array

```c++
class Solution {
public:
    int minimumMountainRemovals(vector<int>& nums) {
        int n = nums.size();
        vector<int> inc,dec;
        vector<int> finc(n), fdec(n);
        for(int i =0;i<n; ++ i ){
            auto idx = lower_bound(inc.begin(), inc.end(), nums[i])-inc.begin();
            if(idx == inc.size()){
                inc.push_back(nums[i]);
            }else {
                inc[idx] = nums[i];
            }
            finc[i] = idx+1;
        }
        for(int i =n-1;i>=0; --i ){
            auto idx = lower_bound(dec.begin(), dec.end(), nums[i])-dec.begin();
            if(idx == dec.size()){
                dec.push_back(nums[i]);
            }else {
                dec[idx] = nums[i];
            }
            fdec[i] = idx+1;
        }
        int ans = 0;
        for(int i = 1;i<n-1;++i){
            if(finc[i]!=1 && fdec[i]!=1)
            ans = max(ans,finc[i] + fdec[i]-1);
        }
        return n-ans;
    }
};
```

### russian-doll-envelopes

```c++
class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        int n = envelopes.size();
        sort(envelopes.begin(),envelopes.end(),[](vector<int> &lhs,vector<int>rhs){
            if(lhs[0] == rhs[0]) return lhs[1] > rhs[1];
            else return lhs[0] < rhs[0];
            });
            vector<int> g;
       for(int i = 0;i<n;++i){
           auto it = lower_bound(g.begin(), g.end(), envelopes[i][1]);
           if(it == g.end()){
               g.push_back(envelopes[i][1]);
           }else {
               *it= envelopes[i][1];
           }
       } 
        return g.size();
    }
};
```

### best-team-with-no-conflicts

```c++
class Solution {
public:
    int bestTeamScore(vector<int>& scores, vector<int>& ages) {
       int n = scores.size();
       pair<int,int> a[n];
       for(int i = 0;i<n;++i){
           a[i] = {scores[i],ages[i]};
       }
       sort(a,a+n);
       int f[n];memset(f, 0, sizeof(f));
       for(int i = 0;i<n;++i){
           for(int j = 0;j<i;++j){
               if(a[j].second <= a[i].second) {
                   f[i] = max(f[i],f[j]);
               }
           }
          f[i] += a[i].first;
       }
       return *max_element(f, f+n);
    }

};
```

## group two

### best-time-to-buy-and-sell-stock

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int inf = 1e9;
        int minprice = inf,maxp = 0;
        for(int p:prices) {
            maxp = max(maxp,p-minprice);
            minprice = min(p,minprice);
        }
        return maxp;
    }
};
```

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n  = prices.size();
        return dfs(prices,n-1,false); 
    }
    int dfs(vector<int> &prices,int i, bool hold) {
        if(i < 0) {
            if(hold) return -1e8;
            else return 0;
        }
        if(hold) {
            return max(dfs(prices,i-1,true),dfs(prices,i-1,false)-prices[i]);
        }
        return max(dfs(prices,i-1,false),dfs(prices,i-1,true)+prices[i]);
    }
};
// dp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n  = prices.size();
        int f[n+1][2] ;
        for(int i = 0;i<=n;++i ) {
            for(int j = 0;j<2;++j) {
                f[i][j] = 0;
            }
        }
        f[0][1] = -1e8;
        for(int i = 0; i < n; ++ i ){
            f[i+1][0] = max(f[i][0],f[i][1]+prices[i]);
           f[i+1][1] = max(f[i][1],f[i][0]-prices[i]); 
        }
        return f[n][0]; 
    }
};
// opt
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n  = prices.size();
        int f0 = 0;
        int f1 = -1e8;
        int nf0;
        for(auto v : prices) {
           nf0 = max(f0,f1+v);
            f1 = max(f1,f0-v);
            f0 = nf0;
        }
        return f0;
    }
};

```

### best-time-to-buy-and-sell-stock-with-cooldown

```c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
       int n = prices.size();
       int f[n+2][2] ;
      for(int i = 0;i<n+2;++i) {
          for(int j = 0;j<2;++j){
              f[i][j] = 0;
          }
      } 
      f[0][1] = -1e8;
      f[1][1] = -1e8;
       for(int i =0 ;i<n;i++ ){
          f[i+2][1] = max(f[i+1][1],f[i][0]-prices[i]);
          f[i+2][0] = max(f[i+1][0],f[i+1][1] + prices[i]);
       }
       return f[n+1][0];
    }
};
//opt
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int p = 0;
        int f0 = 0;
        int f1 = -1e8; 
        for(int i =0 ;i<n;i++ ){
            int tmpf1 = max(f1, p - prices[i]);
            int tmpf0 = max(f0, f1 + prices[i]);
            int tmpp = f0;
            f1 = tmpf1;
            f0 = tmpf0;
            p = tmpp; 
        }
        return f0;
    }
};
```

### best-time-to-buy-and-sell-stock-iv

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        @cache
        def dfs(i,j,hold):
            if j < 0:
                return -inf
            if i < 0:
               return -inf if hold else 0 
            if hold :
                return max(dfs(i-1,j,True), dfs(i-1,j-1,False)-prices[i])
            return max(dfs(i-1,j,False), dfs(i-1,j,True) + prices[i])
        return dfs(n-1,k,False)
```

### best-time-to-buy-and-sell-stock-iii/

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        f = [[-inf] * 2 for _ in range(4)]
        for j in range(1,4):
            f[j][0] = 0
        for i,p in enumerate(prices):
            for j in range(3,0,-1):
                f[j][1] = max(f[j][1],f[j][0] - p)
                f[j][0] = max(f[j][0],f[j-1][1] + p)
        return f[3][0]

```

## group threee

1000
longest-palindromic-subsequence

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        @cache
        def dfs(i,j):
            if i>j:
                return 0
            if i==j: 
                return 1
            if s[i] == s[j]:
                return dfs(i+1,j-1) + 2
            return max(dfs(i+1,j),dfs(i,j-1))
        return dfs(0,n-1)
// dp
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        f = [[0] * n for _ in range(n)]
        for i in range(n-1,-1,-1):
            f[i][i] = 1
            for j in range(i+1,n):
                if s[i] == s[j]:
                    f[i][j] = f[i+1][j-1]+2
                else:
                    f[i][j] = max(f[i][j-1],f[i+1][j])
        return f[0][n-1]
```

### minimum-score-triangulation-of-polygon

```python
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        n  = len(values)
        
        @cache
        def dfs(i,j):
            if i + 1 == j:
                return 0
            res = inf
            for k in range(i+1,j):
                res = min(res, dfs(i,k) + dfs(k,j) + values[i] * values[j] * values[k])
            return res

        return dfs(0,n-1)
```

### guess-number-higher-or-lower-ii

```python
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        @cache
        def dfs(i,j):
            if i>=j:
                return 0
            res = inf
            for k in range(i,j+1):
                res = min(res,max(dfs(i,k-1),dfs(k+1,j))+k)
            return res 
        return dfs(1,n)
```

### minimum-insertion-steps-to-make-a-string-palindrome

```python
class Solution:
    def minInsertions(self, s: str) -> int:
        n  = len(s)
        @cache
        def dfs(i,j):
            if i>j:
                return 0
            if i==j:
                return 1
            if s[i] == s[j]:
                return dfs(i+1,j-1)+2
            return max(dfs(i,j-1),dfs(i+1,j))
        return n-dfs(0,n-1)
```

### maximize-palindrome-length-from-subsequences

```python
class Solution:
    def longestPalindrome(self, word1: str, word2: str) -> int:
        s = word1 + word2
        n = len(s)
        ans = 0
        @cache
        def dfs(i,j):
            if i>j:return 0
            if i == j:return 1
            if s[i] == s[j]:
                res = dfs(i+1,j-1)+2
                if i < len(word1) <= j:
                    nonlocal ans
                    ans = max(ans, res)
                return res
            return max(dfs(i+1,j),dfs(i,j-1))
        dfs(0,n-1)
        return ans
```

### minimum-cost-to-cut-a-stick

```python
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        @cache
        def dfs(i,j):
            if i+1 >= j:
                return 0
            res = inf
            for k in cuts:
                if i < k < j:
                    res = min(res,dfs(i,k)+dfs(k,j)+j-i)
            return res if res != inf else 0
        return dfs(0,n)
//  
class Solution:
    def minCost(self, n: int, cuts: List[int]) -> int:
        cuts = [0] + sorted(cuts) + [n]
        def dfs(i,j):
            if i +1 >= j: 
                return 0
            res = inf
            for k in range(i+1,j):
                res = min(res,dfs(i,k)+dfs(k,j) + cuts[j]-cuts[i])
            return res 
        return dfs(0,len(cuts)-1)


```

### minimum-cost-to-merge-stones

```python
class Solution:
    def mergeStones(self, stones: List[int], k: int) -> int:
        n = len(stones)
        if (n-1) % (k-1) != 0:
            return -1
        s = list(accumulate(stones,initial=0))
        @cache
        def dfs(i,j,p):
            if p == 1:
                return 0 if i==j else dfs(i,j,k) + s[j+1]-s[i]
            res = inf
            for m in range(i,j,k-1):
                res = min(res, dfs(i,m,1) + dfs(m+1,j,p-1))
            return res
        return dfs(0,n-1,1)
```
