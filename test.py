import psutil

temps = psutil.sensors_temperatures() 

cpu_temp = temps["coretemp"][0].current # Get CPU temp
print(f"CPU Temperature: {cpu_temp}°C")  