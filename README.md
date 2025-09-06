# HLFMA 1/5

### 의존성 패키지 다운로드 (rosdep)
```
rosdep install --from-paths . --ignore-src -r -y
```

### git submodule 가져오기
```
git submodule update --recursive
```

### git submodule 업데이트 하기 (tensorrtx)
```
cd path/to/ieve_2025/src/vision/src/tensorrtx
git fetch origin
git checkout ieve_2025/PC
git pull origin ieve_2025/PC
```
