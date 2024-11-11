using BenchmarkTools

function format_num(n)
  units = ["ns", "\$\\mu\$s", "ms", "s"]
  exp_div3 = trunc(Int, trunc(Int, log10(n))/3)

  num_sized = n/exp10(exp_div3*3)
  return string(num_sized)[1:7] * units[exp_div3]
end


function measure_func(func, args)
  bench = @benchmark $func(($args)...)
  return (
      worst = maximum(bench.times),
      best = minimum(bench.times),
      mean = mean(bench.times)
  )
end