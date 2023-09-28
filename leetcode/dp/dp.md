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

1671
354
1626

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
minimum-number-of-removals-to-make-mountain-array
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
