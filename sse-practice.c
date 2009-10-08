
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <xmmintrin.h>
#include <sys/time.h>

static void
inspectLH(__m128 l, __m128 h)
{
    float f1[4];
    float f2[4];
    _mm_storeu_ps(f1, l);
    _mm_storeu_ps(f2, h);
    
    printf("[L:%f H:%f] [L:%f H:%f] [L:%f H:%f] [L:%f H:%f] \n", f1[0], f2[0], f1[1], f2[1], f1[2], f2[2], f1[3], f2[3]);
}

static void inline
inspect(__m128 v)
{
    float f[4];
    _mm_storeu_ps(f, v);
    printf("[%f,%f,%f,%f]\n", f[0], f[1], f[2], f[3]);
}

#define COMPARE_TWO(x,y)                        \
    min = _mm_min_ps(x, y);                     \
    max = _mm_max_ps(x, y);                     \
    x = min; y = max;
#define odd_merge_in_register_sort(x0, x1, x2, x3)              \
    {                                                           \
        __m128 min, max;                                        \
        __m128 tmp;                                             \
        __m128 row0, row1;                                      \
                                                                \
                                                                \
        COMPARE_TWO(x0, x1);                                    \
        COMPARE_TWO(x2, x3);                                    \
                                                                \
        COMPARE_TWO(x1, x2);                                    \
                                                                \
        COMPARE_TWO(x0, x1);                                    \
        COMPARE_TWO(x2, x3);                                    \
                                                                \
        COMPARE_TWO(x1, x2);                                    \
                                                                \
        tmp = _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(1,0,1,0));     \
        row1 = _mm_shuffle_ps(x2, x3, _MM_SHUFFLE(1,0,1,0));    \
                                                                \
        row0 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(2,0,2,0)); \
        row1 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(3,1,3,1)); \
                                                                \
        tmp = _mm_shuffle_ps(x0, x1, _MM_SHUFFLE(3,2,3,2));     \
        x0 = row0; x1 = row1;                                   \
        row1 = _mm_shuffle_ps(x2, x3, _MM_SHUFFLE(3,2,3,2));    \
                                                                \
        row0 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(2,0,2,0)); \
        row1 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(3,1,3,1)); \
        x2 = row0; x3 = row1;                                   \
                                                                \
    }


#define bitonic_merge_kernel(a, b)                      \
{                                                       \
    __m128 l, h, lp, hp, ol, oh;                        \
    b = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0,1,2,3));  \
                                                        \
    l = _mm_min_ps(a, b);                             \
    h = _mm_max_ps(a, b);                             \
    lp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(1,0,1,0));    \
    hp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(3,2,3,2));    \
                                                        \
    l = _mm_min_ps(lp, hp);                             \
    h = _mm_max_ps(lp, hp);                             \
    lp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(2,0,2,0));    \
    lp = _mm_shuffle_ps(lp, lp, _MM_SHUFFLE(3,1,2,0));  \
    hp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(3,1,3,1));    \
    hp = _mm_shuffle_ps(hp, hp, _MM_SHUFFLE(3,1,2,0));  \
                                                        \
    l = _mm_min_ps(lp, hp);                             \
    h = _mm_max_ps(lp, hp);                             \
    ol = _mm_shuffle_ps(l, h, _MM_SHUFFLE(1,0,1,0));    \
    oh = _mm_shuffle_ps(l, h, _MM_SHUFFLE(3,2,3,2));    \
                                                        \
    a = _mm_shuffle_ps(ol, ol, _MM_SHUFFLE(3,1,2,0));  \
    b = _mm_shuffle_ps(oh, oh, _MM_SHUFFLE(3,1,2,0));  \
}

// assume the length of list is 16
static void
bitonic_sort_16elems(float *ret, float* list)
{
    __m128 x[4];
    int i;

    for(i = 0;i < 4;i++){
        x[i] = _mm_load_ps(list + 4 * i);
    }
    odd_merge_in_register_sort(x[0], x[1], x[2], x[3]);
    bitonic_merge_kernel(x[0], x[1]);
    bitonic_merge_kernel(x[2], x[3]);
    // puts("bitonic_sort_16elems:");
    // inspect(x[0]); inspect(x[1]); inspect(x[2]); inspect(x[3]);
    // puts("----");
    _mm_storeu_ps(ret, x[0]);
    _mm_storeu_ps(ret+4, x[1]);
    _mm_storeu_ps(ret+8, x[2]);
    _mm_storeu_ps(ret+12, x[3]);
}

/* list must be 16 aligned */
/* for now, length is assumed to be 2^n */
static void
merge_sort(float *buffer, float *list, uintptr_t length);
static void
merge_sort_rev(float *buffer, float *list, uintptr_t length);
static void inline
merge_sort_merge(float *output, float *input, uintptr_t length);

/* merge_sort is expected to sort list in place */
static void
merge_sort(float *buffer, float *list, uintptr_t length)
{
    uintptr_t half = length / 2;
    if (length == 16){
        bitonic_sort_16elems(buffer, list);
    } else {
        merge_sort_rev(buffer, list, half);
        merge_sort_rev(buffer + half, list + half, half);
    }
    merge_sort_merge(list, buffer, length);
}

/* merge_sort_rev is expected to sort list and write output to buffer */
static void 
merge_sort_rev(float *buffer, float *list, uintptr_t length)
{
    uintptr_t half = length / 2;
    if (length == 16){
        bitonic_sort_16elems(list, list);
    } else {
        merge_sort(buffer, list, half);
        merge_sort(buffer + half, list + half, half);
    }
    
    merge_sort_merge(buffer, list, length);
}

static void inline
merge_sort_merge(float *output, float *input, uintptr_t length)
{

    uintptr_t half;
    float *halfptr;
    float *sentinelptr;

    half = length / 2;
    halfptr = input + half;
    sentinelptr = input + length;

    __m128 x, y;
    float *list1 = input;
    float *list2 = halfptr;
    x = _mm_load_ps(list1);
    y = _mm_load_ps(list2);
    list1 += 4;
    list2 += 4;
    bitonic_merge_kernel(x, y);
    _mm_storeu_ps(output, x);
    output += 4;

    while(true){
        if (*list1 < *list2){
            x = _mm_load_ps(list1);
            list1 += 4;
            bitonic_merge_kernel(x, y);
            _mm_storeu_ps(output, x);
            output += 4;
            if (list1 >= halfptr){
                goto nomore_in_list1;
            }
        } else {
            x = _mm_load_ps(list2);
            list2 += 4;
            bitonic_merge_kernel(x, y);
            _mm_storeu_ps(output, x);
            output += 4;
            if (list2 >= sentinelptr){
                goto nomore_in_list2;
            }
        }
    }
nomore_in_list1:
    while(list2 < sentinelptr){
        x = _mm_load_ps(list2);
        list2 += 4;
        bitonic_merge_kernel(x, y);
        _mm_storeu_ps(output, x);
        output += 4;
    }
    goto end;
nomore_in_list2:
    while(list1 < halfptr){
        x = _mm_load_ps(list1);
        list1 += 4;
        bitonic_merge_kernel(x, y);
        _mm_storeu_ps(output, x);
        output += 4;
    }
end:
    
    _mm_storeu_ps(output, y);
    return;
}

static void
merge_sort_test(uintptr_t num_exp)
{
    float *list;
    float *buffer;
    uintptr_t num = 1 << num_exp;
    struct timeval start_time, end_time;
    
    fprintf(stderr, "size: %ld.\n", num);
    posix_memalign(&list, 16, sizeof(float) * num);
    posix_memalign(&buffer, 16, sizeof(float) * num);

    memset(list, 0, sizeof(float) * num);
    memset(buffer, 0, sizeof(float) * num);

    uintptr_t i;
    srand(0);
    for(i = 0;i < num;i++){
        list[i] = (float)drand48();
    }
    fprintf(stderr, "number generated.\n");
    
    gettimeofday(&start_time, NULL);
    merge_sort(buffer, list, num);
    gettimeofday(&end_time, NULL);
    
    fprintf(stderr, "sorted.\n");
    for(i = 0;i < num - 1;i++){
        if (list[i] > list[i+1]){
            printf("[%ld]%f, [%ld]%f: not sorted.\n",
                   i, list[i], i+1, list[i+1]);
        }
    }
    fprintf(stderr, "checked.\n");

    printf("time: %lf sec.\n", (double)(end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/1000000);

    for(i = 0;i < num;i++){
        // printf("%f\n", list[i]);
    }
}

int main(int argc, char **argv)
{
    uintptr_t num_exp = 10;
    if (argc >= 2){
        num_exp = atoi(argv[1]);
    }
    
    merge_sort_test(num_exp);
    
    return 0;
}

