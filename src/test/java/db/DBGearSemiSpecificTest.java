package test.java.db;

import main.java.db.DBGear;
import main.java.input.CParameter;
import org.testng.Assert;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.util.Set;

public class DBGearSemiSpecificTest {

    @BeforeMethod
    public void configureDigest() {
        DBGear.init_enzymes();
        CParameter.enzyme = 4; // Arg-C without the Pro restriction
        CParameter.maxMissedCleavages = 0;
        CParameter.minPeptideLength = 4;
        CParameter.maxPeptideLength = 8;
        CParameter.clip_nTerm_M = false;
        CParameter.semiSpecific = false;
    }

    @Test
    public void semiSpecificDigestAddsOnlyPeptidesWithAnEnzymaticTerminus() {
        DBGear dbGear = new DBGear();
        String sequence = "AAAAARBBBBBRCCCCC";

        Set<String> specific = dbGear.digest_protein(DBGear.getEnzymeByIndex(CParameter.enzyme), sequence);
        Assert.assertTrue(specific.contains("AAAAAR"));
        Assert.assertFalse(specific.contains("AAAR"));

        CParameter.semiSpecific = true;
        Set<String> semiSpecific = dbGear.digest_protein(DBGear.getEnzymeByIndex(CParameter.enzyme), sequence);
        Assert.assertTrue(semiSpecific.contains("AAAAAR"), "Fully specific peptides remain included.");
        Assert.assertTrue(semiSpecific.contains("AAAAA"), "Protein N-terminus is an enzymatic terminus.");
        Assert.assertTrue(semiSpecific.contains("AAAR"), "A peptide ending at an Arg-C site is allowed.");
        Assert.assertFalse(semiSpecific.contains("AAARB"), "A peptide with two non-enzymatic termini is excluded.");
    }
}
