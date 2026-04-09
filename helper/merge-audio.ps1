param(
    [Parameter(Mandatory)][string]$Visual,  # .mp4 or .jpg
    [Parameter(Mandatory)][string]$Audio,   # .m4a
    [string]$Output = "output.mp4"
)
 
$isImage = $Visual -match '\.(jpg|jpeg|png|JPEG|JPG|PNG)$'
 
if ($isImage) {
    # Get audio duration
    $dur = & ffprobe -v error -show_entries format=duration -of csv=p=0 $Audio
    ffmpeg -loop 1 -i $Visual -i $Audio -c:v libx264 -tune stillimage -c:a aac -b:a 192k -shortest -t $dur $Output
} else {
    # Get durations and use the longer one
    $vDur = [float](& ffprobe -v error -show_entries format=duration -of csv=p=0 $Visual)
    $aDur = [float](& ffprobe -v error -show_entries format=duration -of csv=p=0 $Audio)
    $dur  = [Math]::Max($vDur, $aDur)
    ffmpeg -i $Visual -i $Audio -c:v copy -c:a aac -b:a 192k -map 0:v:0 -map 1:a:0 -t $dur $Output
}
 
Write-Host "Done $Output"

# Example usage:
#.\merge-audio.ps1 -Visual clip.mp4 -Audio track.m4a
#.\merge-audio.ps1 -Visual photo.jpg -Audio track.m4a -Output result.mp4
