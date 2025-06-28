#ifndef __BASIC_TESTS_DEFS__
#define __BASIC_TESTS_DEFS__

#define BEGIN_TEST() \
    try              \
    {

#define END_TEST()                                      \
    }                                                   \
    catch (const std::exception &ex)                    \
    {                                                   \
        FAIL() << "Caught exception: " << ex.what();    \
    }                                                   \
    catch (...)                                         \
    {                                                   \
        FAIL() << "Caught unknown exception";           \
    }

#define SAFE_TEST(code_to_execute, handler_code)        \
    try {                                               \
        code_to_execute;                                \
    } catch (const std::exception& e) {                 \
        handler_code;                                   \
    } catch (...) {                                     \
        FAIL() << "Caught unknown exception";           \
    }

#endif