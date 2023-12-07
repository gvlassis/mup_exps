# The number of processes launched in parallel. Assuming ImageMagick uses 1 core, "num_processes" should be equal to the number of available threads
num_processes="${1}"
# The path of the CLS-LOC directory, which contains the images under train/, val/ and test/
CLS_LOC_path="${2}"
# Images interval between updates of _resize()
update_frequency="${3}"
# The final height of the resized images
height="${4}"

images_paths=($(find "${CLS_LOC_path}" -name "*.JPEG"))
num_images=${#images_paths[*]}
printf "📷 There are \x1b[36m$num_images\x1b[0m images\n"

function _resize(){
	local subimages_paths=(${@})
	local num_subimages=${#subimages_paths[*]}

	local image
	for ((image=0; image<num_subimages; image++)); do

		convert "${subimages_paths[$image]}" -type truecolor -resize "${height}x${height}" -background black -gravity center -extent "${height}x${height}" "${subimages_paths[$image]}"

		if (( $image % $update_frequency == 0)); then
			printf "🍻 \x1b[96m$image/$((num_subimages-1))\x1b[0m\n"
		fi
	done
}

quotient="$((num_images/num_processes))"
remainder="$((num_images%num_processes))"
left_index=0
for ((process=0; process<num_processes; process++)); do
	num_subimages="$((process<remainder ? quotient+1:quotient))"
	subimages_paths=(${images_paths[@]:$left_index:$num_subimages})
	_resize ${subimages_paths[@]} &
	printf "🧵 Process $process launched\n"
	left_index="$((left_index+num_subimages))"
done

wait

printf "🍾 Done!"