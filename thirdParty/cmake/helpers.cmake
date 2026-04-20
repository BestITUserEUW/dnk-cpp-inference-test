
function(get_git_version OUTPUT_VAR)
    # Find Git executable
    find_program(GIT_EXECUTABLE git)

    if(NOT GIT_EXECUTABLE)
        message(STATUS "Git not found. Falling back to empty version number")
        set(VERSION_STR "0.0.0")
    else()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --abbrev=7 --always --tags
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE VERSION_STR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE GITVER
        )

        if(NOT GITVER EQUAL "0")
            message(STATUS "No git version found. Falling back to empty version number")
            set(VERSION_STR "0.0.0")
        endif()
    endif()

    # Set the output variable
    set(${OUTPUT_VAR} "${VERSION_STR}" PARENT_SCOPE)
endfunction()