using Pkg

println("\n--- 1. Ensuring dependencies are in project ---")
Pkg.activate(".")
# We must add all packages that model_core.jl depends on, including PythonCall
Pkg.add(["PackageCompiler", "JuMP", "HiGHS", "PythonCall", "LinearAlgebra", "SparseArrays", "Random", "Statistics"])

using PackageCompiler

# 1. Identify where we are
SCRIPTS_DIR = dirname(@__FILE__)
SYSIMAGE_PATH = joinpath(SCRIPTS_DIR, "sysimage.so")
JULIA_CORE = joinpath(SCRIPTS_DIR, "model_core.jl")

println("\n--- 2. Starting Sysimage Build (this takes 5-10 minutes) ---")
println("Target Path: ", SYSIMAGE_PATH)

# We include these packages in the sysimage for instant loading.
# Including ModelCore ensures the solver logic is also pre-compiled.
create_sysimage(
    [:LinearAlgebra, :SparseArrays, :Random, :Statistics, :JuMP, :HiGHS, :PythonCall],
    sysimage_path=SYSIMAGE_PATH,
    precompile_execution_file=JULIA_CORE
)

println("\n--- 3. Build Complete! ---")
println("The file 'Scripts/sysimage.so' has been created.")
println("\nTo use it automatically, ensure your Python scripts set:")
println("os.environ[\"PYTHON_JULIACALL_SYSIMAGE\"] = \"$(SYSIMAGE_PATH)\"")
println("before importing any other packages.")
