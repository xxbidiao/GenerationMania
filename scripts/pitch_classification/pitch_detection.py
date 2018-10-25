def get_pitch(filename):
    import sys
    from aubio import source, pitch
    from pydub import AudioSegment
    song = AudioSegment.from_file(filename)
    downsample = 1
    samplerate = song.frame_rate // downsample

    win_s = 4096 // downsample # fft size
    hop_s = 512  // downsample # hop size

    s = source(filename, samplerate, hop_s)
    samplerate = s.samplerate

    tolerance = 0.8

    pitch_o = pitch("yin", win_s, hop_s, samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    pitches = []
    confidences = []

    # total number of frames read
    total_frames = 0
    check_frames = 0
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        #pitch = int(round(pitch))
        confidence = pitch_o.get_confidence()
        if confidence >= 0:
            pitches += [pitch]
            confidences += [confidence]
            total_frames += read
            if read < hop_s: break
        else:
            check_frames += 1
            if check_frames > 128: break

    if (len(pitches) != 0):
        return(sum(pitches) / float(len(pitches)))
    else:
        print("Error determining pitch. Closest estimate is " + str(pitch))
        return float(pitch)
