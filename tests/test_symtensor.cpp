
#include <detail/dense/symtensor.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace symprop;

TEST_CASE( "Symmetric tensor size", "[SymtensorBase]" ) {
    Symtensor<2> symtensor(3);
    REQUIRE( symprop::symtens_size(3, 2) == 6 );
    REQUIRE( symtensor.dim() == 3 );
    REQUIRE( symtensor.order() == 2 );
    REQUIRE( symtensor.packed_size() == 6 );

    Symtensor<2> symtensor2(4);
    REQUIRE( symtensor2.packed_size() == 10 );

    Symtensor<3> symtensor3(3);
    REQUIRE( symtensor3.packed_size() == 10 );

    Symtensor<5> symtensor4(5);
    REQUIRE( symtensor4.packed_size() == 126 );
}

TEST_CASE( "Symmetric tensor outer product", "[Symtensor]" ) {
    std::vector<double> vec1 = {1, 2, 3};
    Symtensor<1> symtensor(vec1);
    std::vector<double> vec2 = {4, 5, 6};

    Symtensor<2> symtensor2(3);
    symtensor2.sym_outer_prod(symtensor, vec2);

    for (dim_t i = 0; i < 3; i++) {
        for (dim_t j = i; j < 3; j++) {
            REQUIRE( symtensor2.at({i, j}) == vec1[i] * vec2[j] );
        }
    }

    std::vector<double> vec3 = {7, 8, 9};
    Symtensor<3> symtensor3(3);
    symtensor3.sym_outer_prod(symtensor2, vec3);

    std::array<dim_t, 3> indices;
    index_t cnt = 0;
    for (dim_t i = 0; i < 3; i++) {
        for (dim_t j = i; j < 3; j++) {
            for (dim_t k = j; k < 3; k++) {
                REQUIRE( symtensor3.at({i, j, k}) == vec1[i] * vec2[j] * vec3[k] );
                REQUIRE( symtensor3[cnt] == vec1[i] * vec2[j] * vec3[k] );
                symtensor3.indices(cnt, indices);
                REQUIRE( indices[0] == i );
                REQUIRE( indices[1] == j );
                REQUIRE( indices[2] == k );
                cnt += 1;
            }
        }
    }
}
