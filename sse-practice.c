
#include <stdio.h>
#include <xmmintrin.h>

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

static void inline
odd_merge_in_register_sort(__m128 *x0, __m128 *x1, __m128 *x2, __m128 *x3)
{
    __m128 min, max;
#define COMPARE_TWO(x,y)                        \
    min = _mm_min_ps(x, y);                     \
    max = _mm_max_ps(x, y);                     \
    x = min; y = max;

    COMPARE_TWO(*x0, *x1);
    COMPARE_TWO(*x2, *x3);

    COMPARE_TWO(*x1, *x2);

    COMPARE_TWO(*x0, *x1);
    COMPARE_TWO(*x2, *x3);

    COMPARE_TWO(*x1, *x2);

    __m128 tmp;
    __m128 row0, row1;
    tmp = _mm_shuffle_ps(*x0, *x1, _MM_SHUFFLE(1,0,1,0));
    row1 = _mm_shuffle_ps(*x2, *x3, _MM_SHUFFLE(1,0,1,0));

    row0 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(2,0,2,0));
    row1 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(3,1,3,1));

    tmp = _mm_shuffle_ps(*x0, *x1, _MM_SHUFFLE(3,2,3,2));
    *x0 = row0; *x1 = row1;
    row1 = _mm_shuffle_ps(*x2, *x3, _MM_SHUFFLE(3,2,3,2));
    
    row0 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(2,0,2,0));
    row1 = _mm_shuffle_ps(tmp, row1, _MM_SHUFFLE(3,1,3,1));
    *x2 = row0; *x3 = row1;

    // puts("odd_merge:");
    // inspect(*x0); inspect(*x1); inspect(*x2); inspect(*x3);
}

static void inline
bitonic_merge_kernel(__m128 *a, __m128 *b)
{
    __m128 l, h, lp, hp, ol, oh;
    
    *b = _mm_shuffle_ps(*b, *b, _MM_SHUFFLE(0,1,2,3));
    inspectLH(*a, *b);

    l = _mm_min_ps(*a, *b);
    h = _mm_max_ps(*a, *b);
    lp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(1,0,1,0));
    hp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(3,2,3,2));
    inspectLH(l, h);
    inspectLH(lp, hp);
    
    l = _mm_min_ps(lp, hp);
    h = _mm_max_ps(lp, hp);
    lp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(2,0,2,0));
    lp = _mm_shuffle_ps(lp, lp, _MM_SHUFFLE(3,1,2,0));
    hp = _mm_shuffle_ps(l, h, _MM_SHUFFLE(3,1,3,1));
    hp = _mm_shuffle_ps(hp, hp, _MM_SHUFFLE(3,1,2,0));
    inspectLH(l, h);
    inspectLH(lp, hp);

    l = _mm_min_ps(lp, hp);
    h = _mm_max_ps(lp, hp);
    ol = _mm_shuffle_ps(l, h, _MM_SHUFFLE(1,0,1,0));
    oh = _mm_shuffle_ps(l, h, _MM_SHUFFLE(3,2,3,2));
    
    *a = _mm_shuffle_ps(ol, ol, _MM_SHUFFLE(3,1,2,0));
    *b = _mm_shuffle_ps(oh, oh, _MM_SHUFFLE(3,1,2,0));
    inspectLH(l, h);
    inspectLH(*a, *b);
}

static void
simple_bitonic_test(void)
{
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[] = {1.1f, 1.2f, 1.3f, 1.4f};

    __m128 a = _mm_set_ps(x[0], x[1], x[2], x[3]);
    __m128 b = _mm_set_ps(y[0], y[1], y[2], y[3]);

    float ret[8];
    bitonic_merge_kernel(&a, &b);
    int i;
    _mm_storeu_ps(ret, a);
    _mm_storeu_ps(ret + 4, b);
    for(i = 0;i < 8;i++){
        printf("%f ", ret[i]);
    }
    puts("");
}

// assume the length of list is 16
static void
bitonic_sort_two_4x4(float* list)
{
    __m128 x[4];
    int i;

    for(i = 0;i < 4;i++){
        x[i] = _mm_set_ps(list[4*i + 0], list[4*i + 1],
                          list[4*i + 2], list[4*i + 3]);
    }
    odd_merge_in_register_sort(&x[0], &x[1], &x[2], &x[3]);
    puts("odd_merged:");
    inspect(x[0]); inspect(x[1]); inspect(x[2]); inspect(x[3]);
    puts("----");
    bitonic_merge_kernel(&x[0], &x[1]);
    bitonic_merge_kernel(&x[2], &x[3]);
    _mm_storeu_ps(list, x[0]);
    _mm_storeu_ps(list+4, x[1]);
    _mm_storeu_ps(list+8, x[2]);
    _mm_storeu_ps(list+12, x[3]);
}

static void
simple_bitonic_sort_test(void)
{
    float f[] = {12.0f, 21.0f, 4.0f, 13.0f,
                 9.0f, 8.0f, 6.0f, 7.0f,
                 1.0f, 14.0f, 3.0f, 0.0f,
                 5.0f, 11.0f, 15.0f, 10.0f};
    int i;

    bitonic_sort_two_4x4(f);
    for(i = 0;i < 8;i++){
        printf("%f ", f[i]);
    }
    puts("");
    for(i = 8;i < 16;i++){
        printf("%f ", f[i]);
    }
    puts("");
}

int main(void)
{
    simple_bitonic_sort_test();
    
    return 0;
}

